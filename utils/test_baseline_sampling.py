import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from datasets import Dataset

from baseline_hard_negs import build, draw
from preprocess_images import choose_negative, save_processed_dataset
from preprocess_text import load_esci_dataset
from utils.paper_analysis import active_dataset_bases, parse_run_name
from train import build_pair_dataset
from utils.distance_labels import to_training_labels
from utils.esci_candidates import eligible_query_ids, filter_processed_dataset, filter_raw_pairs
from utils.image_split import garment_id
from utils.leakage_split import build as split_raw


def text_row(query, negative, source="Substitute", side="train", positive="p"):
    return dict(original_query=query, nl_query=f"{query}-{negative}",
                rephrased_query=f"natural {query}-{negative}", split=side,
                positive_id=positive, negative_id=negative, positive_example=f"doc {positive}",
                negative_example=f"doc {negative}", negative_example_source=source,
                query_distance=None if source == "random" else 2.0)


def easy_twin(row):
    return dict(row, negative_id="easy", negative_example="doc easy",
                negative_example_source="random", query_distance=None)


def raw(row):
    return dict(original_query=row["original_query"], nl_query=row["nl_query"], item="item",
                positive_product={"product_id": row["positive_id"], "product_text": row["positive_example"]},
                hard_neg_product={"product_id": row["negative_id"], "product_text": row["negative_example"]},
                negative_example_source=row["negative_example_source"], query_distance=2)


def image_row(query, positive, negative, category="tees", neg_category="tees", side="train",
              source="same_category2_different_product"):
    return dict(original_query="", nl_query=query, rephrased_query=query, split=side,
                positive_product_id=positive, negative_product_id=negative,
                positive_example=positive + ".jpg", negative_example=negative + ".jpg",
                positive_category=category, negative_category=neg_category,
                negative_example_source=source, query_distance=2.0)


class CandidateTests(unittest.TestCase):
    def test_early_gate_counts_distinct_products_across_both_labels(self):
        data = pd.DataFrame({"query_id": [1, 1, 2, 2, 3, 3],
                             "product_id": ["a", "a", "b", "c", "d", "e"],
                             "esci_label": ["Substitute", "Substitute", "Substitute", "Irrelevant",
                                            "Substitute", "Complement"]})
        self.assertEqual(eligible_query_ids(data), {2})

    def test_loader_filters_before_generation_and_rechecks_language_losses(self):
        records = []
        for query_id, product, label in [(1, "p1", "Exact"), (1, "a1", "Substitute"),
                                         (1, "b1", "Irrelevant"), (2, "p2", "Exact"),
                                         (2, "a2", "Substitute"), (2, "a2", "Substitute"),
                                         (3, "p3", "Exact"), (3, "a3", "Substitute"),
                                         (3, "foreign", "Irrelevant")]:
            records.append(dict(query_id=query_id, query=f"query {query_id}", product_id=product,
                                product_title=product, product_text=product, product_locale="us",
                                esci_label=label))
        with patch("preprocess_text.load_dataset", return_value={"train": Dataset.from_list(records)}), \
             patch("preprocess_text.detect", side_effect=lambda text: "es" if text == "foreign" else "en"):
            rows = load_esci_dataset()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["query"] for r in rows}, {"query 1"})
        self.assertEqual({r["negative_example_source"] for r in rows}, {"Substitute", "Irrelevant"})

    def test_baseline_identity_is_distinct_from_legacy_random_runs(self):
        prefix = "text__mpnet__cosent__catalog_candidates2_rephrased-in-context_"
        suffix = "-rephrased__rephrased__note-paper"
        self.assertEqual(parse_run_name(prefix + "baseline" + suffix)["negs"], "baseline")
        self.assertEqual(parse_run_name(prefix + "baseline-v3" + suffix)["negs"], "baseline-v3")
        self.assertEqual(parse_run_name(prefix + "random" + suffix)["negs"], "random")
        self.assertTrue(active_dataset_bases()["text"].endswith("_candidates2_humanholdout"))

    def test_language_or_generation_failure_removes_entire_singleton_group(self):
        a, b = raw(text_row("q", "a")), raw(text_row("q", "b", "Irrelevant"))
        self.assertEqual(len(filter_raw_pairs([a, b])), 2)
        self.assertEqual(filter_raw_pairs([a]), [])
        self.assertEqual(filter_raw_pairs([a, a]), [])
        b["hard_neg_product"]["product_text"] = a["hard_neg_product"]["product_text"]
        self.assertEqual(filter_raw_pairs([a, b]), [])

    def test_all_splits_filtered_and_easy_twins_follow_comparisons(self):
        a, b = text_row("q", "a"), text_row("q", "b", "Irrelevant")
        singleton = text_row("single", "c")
        test_singleton = text_row("test-single", "d", side="test", positive="test-p")
        rows = [a, easy_twin(a), b, easy_twin(b), singleton, easy_twin(singleton),
                test_singleton, easy_twin(test_singleton)]
        out, report = filter_processed_dataset(Dataset.from_list(rows))
        self.assertEqual(out.to_list(), rows[:4])
        self.assertEqual(report["retained_groups"], 1)

    def test_split_cleanup_rechecks_minimum(self):
        # B has two candidates before splitting; one conflicts with train and is dropped.
        train = [raw(text_row("A", "a")), raw(text_row("A", "b", "Irrelevant"))]
        held = [raw(text_row("B", "a", positive="test-p")),
                raw(text_row("B", "d", "Irrelevant", positive="test-p"))]
        for r in train: r["item"] = "train-item"
        for r in held: r["item"] = "test-item"
        with tempfile.TemporaryDirectory() as temp:
            import json
            path = Path(temp) / "raw.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in train + held))
            with patch("utils.leakage_split.assign_side", side_effect=lambda g, *args: "train" if g == "train-item" else "test"):
                kept, _, _ = split_raw(path, .1, .1, 42)
        self.assertEqual({r["original_query"] for r in kept}, {"A"})

    def test_text_baseline_preserves_easy_and_evaluation_and_uses_combined_pool(self):
        a, b = text_row("q", "a"), text_row("q", "b", "Irrelevant")
        evaluation = text_row("eval", "eval-n", side="test", positive="eval-p")
        data = Dataset.from_list([a, easy_twin(a), b, easy_twin(b), evaluation])
        out, report, records = build(data, "text", "rephrased")
        self.assertEqual(out[1], data[1])
        self.assertEqual(out[3], data[3])
        self.assertEqual(out[4], data[4])
        self.assertEqual(report["min_candidate_count"], 2)
        self.assertTrue(all(r["mined_id"] in {"a", "b"} for r in records.values()))
        self.assertEqual(out[0]["negative_example_source"], "baseline")
        self.assertIsNone(out[0]["query_distance"])
        self.assertEqual(records, draw(data, "text", "rephrased")[0])
        # Both labels and the original candidate must remain reachable.
        selections = {draw(data, "text", "rephrased", seed)[0][0]["mined_id"] for seed in range(12)}
        self.assertEqual(selections, {"a", "b"})

    def test_v3_excludes_construction_product_and_identical_document_aliases(self):
        a = text_row("q", "a")
        alias = dict(text_row("q", "alias", "Irrelevant"), negative_example=a["negative_example"])
        b = text_row("q", "b", "Irrelevant")
        evaluation = text_row("eval", "eval-n", side="test", positive="eval-p")
        data = Dataset.from_list([a, easy_twin(a), alias, easy_twin(alias), b, easy_twin(b), evaluation])
        out, report, records = build(data, "text", "rephrased", version="v3")
        self.assertEqual(out[0]["negative_id"], "b")
        self.assertEqual(out[2]["negative_id"], "b")
        self.assertEqual(out[4]["negative_example"], a["negative_example"])
        self.assertEqual(report["n_selected_original"], 0)
        self.assertFalse(report["original_negative_eligible"])
        self.assertEqual(report["version"], "v3")
        self.assertEqual(report["min_candidate_count"], 1)
        for index in (1, 3, 5, 6):
            self.assertEqual(out[index], data[index])
        for index in records:
            self.assertNotEqual(out[index]["negative_id"], data[index]["negative_id"])
            self.assertNotEqual(out[index]["negative_example"], data[index]["negative_example"])
            self.assertIsNone(out[index]["query_distance"])
        self.assertEqual(records, draw(data, "text", "rephrased", version="v3")[0])

    def test_baseline_rejects_singleton_instead_of_global_fallback(self):
        data = Dataset.from_list([text_row("q", "a"), text_row("another", "b")])
        with self.assertRaisesRegex(ValueError, "two eligible"):
            draw(data, "text", "rephrased")

    def test_cosent_keeps_positive_and_ungraded_mse_accepts_unmeasured_baseline(self):
        data = Dataset.from_list([
            dict(anchor="q", positive="p", negative="n", negative_example_source="baseline", query_distance=None),
            dict(anchor="q", positive="p", negative="e", negative_example_source="random", query_distance=None)])
        pairs = build_pair_dataset(data, is_cosent=True)
        self.assertEqual(list(pairs["label"]), [1.0, 0.0, 0.0])
        labels, stats = to_training_labels(data, 40, 10, "linear", unmeasured_as_easy=True)
        self.assertEqual(list(labels["label"]), [.25, .25])
        with self.assertRaisesRegex(ValueError, "no measured"):
            to_training_labels(data, 40, 10, "linear")


class ImageNegativeTests(unittest.TestCase):
    def test_views_and_colorways_have_one_garment_identity(self):
        for value in ["MEN_Tees_id_0001_01", "MEN_Tees_id_0001_02", "MEN_Tees_id_0001_02_1_front",
                      "MEN_Tees_id_0001_01_2_back"]:
            self.assertEqual(garment_id(value), "MEN_Tees_id_0001")

    def test_gold_excludes_other_views_and_colorways(self):
        anchor = {"product_id": "a_01", "category2": "tees"}
        candidates = [dict(anchor, product_id=p) for p in ["a_01", "a_02", "a_01_1_front", "b_01"]]
        for seed in range(12):
            chosen = choose_negative(random.Random(seed), candidates, anchor, "category2", True, False)
            self.assertEqual(chosen["product_id"], "b_01")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "view/colorway"):
                save_processed_dataset([{"positive_product_id": "a_01", "hard_negative_product_id": "a_02",
                                         "easy_negative_product_id": "b_01"}], temp + "/out")

    def test_baseline_same_category_different_garment_and_same_shared_easy(self):
        rows = [image_row("q", "a_01", "b_01"),
                image_row("q", "a_01", "e_01", neg_category="skirts", source="random"),
                image_row("other view", "a_02", "c_01"),
                image_row("heldout", "test_01", "held_01", side="test")]
        data = Dataset.from_list(rows)
        for seed in range(12):
            out, _, _ = build(data, "multimodal", "rephrased", seed)
            self.assertIn(out[0]["negative_product_id"], {"b_01", "c_01"})
            self.assertEqual(out[0]["negative_category"], "tees")
            self.assertEqual(out[1], data[1])
            self.assertEqual(out[3], data[3])


if __name__ == "__main__":
    unittest.main()
