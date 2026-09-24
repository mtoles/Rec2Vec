import unittest

from datasets import Dataset

from utils.image_split import verify_image_splits


def row(side, query, positive, negative):
    return {"split": side, "rephrased_query": query,
            "positive_example": positive + '.jpg', "negative_example": negative + '.jpg',
            "positive_product_id": positive, "negative_product_id": negative}


class ImageSplitTests(unittest.TestCase):
    def test_disjoint(self):
        data = Dataset.from_list([row('train', 'red shirt', 'a_01', 'b_01'),
                                  row('test', 'blue shirt', 'c_01', 'd_01')])
        self.assertEqual(verify_image_splits(data), {'train': 1, 'test': 1})

    def test_requires_split(self):
        data = Dataset.from_list([row('train', 'red shirt', 'a_01', 'b_01')]).remove_columns('split')
        with self.assertRaisesRegex(ValueError, 'precomputed'):
            verify_image_splits(data)

    def test_negative_becomes_test_positive(self):
        data = Dataset.from_list([row('train', 'red shirt', 'a_01', 'b_01'),
                                  row('test', 'blue shirt', 'b_01', 'd_01')])
        with self.assertRaisesRegex(ValueError, 'Cross-split image'):
            verify_image_splits(data)

    def test_colorways_share_garment(self):
        data = Dataset.from_list([row('train', 'red shirt', 'a_01', 'b_01'),
                                  row('test', 'blue shirt', 'a_02', 'd_01')])
        with self.assertRaisesRegex(ValueError, 'Cross-split garment'):
            verify_image_splits(data)

    def test_normalized_queries(self):
        data = Dataset.from_list([row('train', 'Red   shirt', 'a_01', 'b_01'),
                                  row('test', ' red shirt ', 'c_01', 'd_01')])
        with self.assertRaisesRegex(ValueError, 'Cross-split query'):
            verify_image_splits(data)


if __name__ == '__main__':
    unittest.main()
