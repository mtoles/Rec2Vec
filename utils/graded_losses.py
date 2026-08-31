"""Batch-graded margin MSE: ours-mse extended over in-batch candidates.

ours-mse (MarginMSELoss) extracts one comparison per row: the margin between the
positive and the row's own hard negative, regressed onto the measured distance.
InfoNCE gets its strength from comparing each anchor against every candidate in the
batch. This loss does both: every candidate in the batch is compared against every
anchor, and every comparison has a distance target -- the measured one for the
anchor's own hard negative, the easy-negative distance for everything else.
"""

import math

import torch
from torch import nn

from sentence_transformers import SentenceTransformer, util


class BatchGradedMarginMSELoss(nn.Module):

    def __init__(self, model: SentenceTransformer, easy_label: float, hard_weight: float = 0.5):
        """
        easy_label: the training label of an unmeasured (random) negative, on the same
            scale as the dataset's label column. With linear transform and V=40 this is
            easy_negative_distance / V = 20/40 = 0.5. Passed in rather than recomputed so
            the loss stays consistent with whatever V / transform the run uses.
        hard_weight: fraction of the loss carried by the measured-distance term. Each
            anchor has 1 graded comparison but 2B-2 easy ones; a plain mean over all
            terms would weight the graded signal 1:(2B-2), burying the thing this loss
            exists to use. 0.5 gives the two kinds of supervision equal say.
        """
        super().__init__()
        self.model = model
        self.easy_label = easy_label
        self.hard_weight = hard_weight

    def forward(self, sentence_features, labels):
        # One feature dict per dataset column, in column order: (anchor, positive,
        # negative). Each tower pass yields pooled embeddings, shape [B, dim]. For
        # multimodal runs the positive/negative features are already image tensors --
        # the collator handled that -- so this works for both modalities.
        anchors, positives, negatives = [
            self.model(features)["sentence_embedding"] for features in sentence_features
        ]
        B = anchors.shape[0]

        # Candidate pool = all positives then all negatives, exactly MNRL's layout:
        #   column j       (j <  B): row j's positive
        #   column B + j:            row j's hard negative
        # One matmul gives every anchor-candidate cosine.
        candidates = torch.cat([positives, negatives], dim=0)      # [2B, dim]
        S = util.cos_sim(anchors, candidates)                      # [B, 2B]

        # Margin of every candidate against the anchor's own positive. S[i, i] is
        # cos(anchor_i, positive_i) because column i < B holds row i's positive.
        # M[i, j] = how much worse candidate j scores than the true positive.
        own_pos = S.diagonal().unsqueeze(1)                        # [B, 1]
        M = own_pos - S                                            # [B, 2B]

        # Target distance for every cell of M:
        #   own positive (column i):    0 -- it is the reference point
        #   own negative (column B+i):  the dataset label (measured d/V; easy-negative
        #                               rows already carry easy_label here)
        #   everything else:            easy_label -- a cross-row candidate is a random
        #                               product w.r.t. this anchor, i.e. an easy negative
        idx = torch.arange(B, device=S.device)
        targets = labels.to(S.dtype)
        D = torch.full_like(S, self.easy_label)
        D[idx, idx] = 0.0
        D[idx, B + idx] = targets

        # Graded term: the one comparison per anchor with a measured distance.
        hard_term = ((M[idx, B + idx] - targets) ** 2).mean()

        # Easy term: all cross-row comparisons. Excluded cells:
        #   - own positive: M and D are both exactly 0 there; including it would only
        #     dilute the mean with guaranteed-zero error terms;
        #   - own hard negative: that is the hard term, already counted.
        # This assumes no other row in the batch shares this anchor's positive. The two
        # dataset rows of one query share anchor AND positive, so a batch holding both
        # would label the twin's positive easy_label when its true distance is 0.
        # train.py always trains this loss under BatchSamplers.NO_DUPLICATES, which
        # guarantees it cannot happen.
        easy_mask = torch.ones_like(S, dtype=torch.bool)
        easy_mask[idx, idx] = False
        easy_mask[idx, B + idx] = False
        easy_cells = (M - D)[easy_mask]
        # A conflict-deferred tail batch can have B=1, leaving no cross-row cells; the
        # mean of an empty tensor is NaN and one NaN backward destroys the model.
        if easy_cells.numel() == 0:
            return hard_term
        easy_term = (easy_cells ** 2).mean()

        return self.hard_weight * hard_term + (1.0 - self.hard_weight) * easy_term


class GradedInfoNCELoss(nn.Module):
    """Softmax cross-entropy over in-batch candidates with distance-graded soft targets.

    infonce-mined with the one-hot target replaced by a per-row distribution built from
    the measured distances. Unnormalized target weight per candidate:
        own positive:        1              (distance 0)
        own MEASURED negative: 1 - label    (label = transformed d/V, so a near-miss
                                             keeps most of its weight, d = V keeps none)
        own random negative: easy_weight    (an unmeasured random product is no more a
                                             match than the cross-row candidates, so it
                                             gets the same weight they do; grading only
                                             ever applies to measured distances)
        cross-row:           easy_weight    (default 0: still in the softmax denominator,
                                             so they are pushed down exactly as in
                                             infonce-mined, but hold no target mass)
    A row's negative is random iff its label equals easy_label -- to_training_labels
    places the easy distance strictly above every measured one. Each row is normalized
    to sum to 1, and the loss is -sum(T * log_softmax(S)). On random-negative rows this
    is exactly infonce-mined; labels are produced by to_training_labels and are always
    in [0, 1].
    """

    def __init__(self, model: SentenceTransformer, easy_label: float,
                 scale: float = 20.0, easy_weight: float = 0.0):
        super().__init__()
        self.model = model
        self.easy_label = easy_label
        self.scale = scale
        self.easy_weight = easy_weight

    def forward(self, sentence_features, labels):
        anchors, positives, negatives = [
            self.model(features)["sentence_embedding"] for features in sentence_features
        ]
        B = anchors.shape[0]

        # Same candidate layout as MNRL and BatchGradedMarginMSELoss:
        # column j < B is row j's positive, column B + j is row j's hard negative.
        candidates = torch.cat([positives, negatives], dim=0)          # [2B, dim]
        S = util.cos_sim(anchors, candidates) * self.scale             # [B, 2B]

        targets = labels.to(S.dtype)
        is_random = targets >= self.easy_label - 1e-6

        idx = torch.arange(B, device=S.device)
        W = torch.full_like(S, self.easy_weight)
        W[idx, idx] = 1.0
        W[idx, B + idx] = torch.where(is_random, torch.full_like(targets, self.easy_weight),
                                      1.0 - targets)
        T = W / W.sum(dim=1, keepdim=True)

        # Shares the twin-positive hazard of BatchGradedMarginMSELoss: a batch holding both
        # rows of one query would give the twin's positive target weight easy_weight when it
        # deserves 1. train.py always trains this loss under NO_DUPLICATES.
        return -(T * torch.log_softmax(S, dim=1)).sum(dim=1).mean()


class GradedSigLIPLoss(nn.Module):
    """Per-pair sigmoid BCE over in-batch candidates with distance-graded soft labels.

    The absolute-placement counterpart of GradedInfoNCELoss: no softmax, no competition.
    Every (anchor, candidate) cell is an independent prediction sigma(s * cos + b) fit to
    its own target similarity:
        own positive:      1
        own hard negative: 1 - label          (label = transformed d/V)
        cross-row:         1 - easy_label     (a random product sits at the easy distance,
                                               the same target as the row's own random
                                               negative -- 0.5 for linear V=40)
    s and b are learnable (the SigLIP recipe); the trainer optimizes loss parameters
    alongside the model. b starts at each mode's prior: -log(2B - 1) in binary mode
    (sigma(b) = 1/2B, the chance a candidate is the match) and 0 in graded mode (most
    cells want 1 - easy_label, and sigma(0) = 0.5 already sits there). At the optimum
    sigma(s*cos+b) = target for every pair. Ranking by sigma(s*cos+b) is ranking by cos,
    so inference is unchanged.
    """

    def __init__(self, model: SentenceTransformer, easy_label: float | None,
                 hard_weight: float = 0.5, init_scale: float = 10.0, init_bias: float | None = None,
                 binary: bool = False, batch_size: int | None = None):
        """
        easy_label: as in BatchGradedMarginMSELoss. Unused in binary mode.
        hard_weight: fraction of the loss carried by the 2B labeled cells (own positive +
            own hard negative). Each row has 2 labeled cells but 2B-2 cross-row ones; a
            plain mean would weight the per-row signal 2:(2B-2). 0.5 gives the two kinds
            of supervision equal say, matching BatchGradedMarginMSELoss.
        binary: the siglip-mined baseline -- identical layout, scale/bias and weighting,
            but one-hot targets (own positive 1, every other cell 0) and no use of the
            label column. The graded/binary comparison then differs in targets only.
        """
        super().__init__()
        assert binary or easy_label is not None
        if init_bias is None:
            if binary:
                # SigLIP's init principle scaled to this batch: sigma(b) = the prior
                # probability that a candidate is the match, 1/2B (Zhai et al. 2023 use
                # b = -10 at |B| ~ 16k by the same logic). Graded targets center at
                # 1 - easy_label instead, and sigma(0) = 0.5 already sits there.
                assert batch_size is not None
                init_bias = -math.log(2 * batch_size - 1)
            else:
                init_bias = 0.0
        self.model = model
        self.easy_label = easy_label
        self.hard_weight = hard_weight
        self.binary = binary
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.logit_bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, sentence_features, labels):
        anchors, positives, negatives = [
            self.model(features)["sentence_embedding"] for features in sentence_features
        ]
        B = anchors.shape[0]

        # Same candidate layout as the other batch-wide losses:
        # column j < B is row j's positive, column B + j is row j's hard negative.
        candidates = torch.cat([positives, negatives], dim=0)          # [2B, dim]
        Z = util.cos_sim(anchors, candidates) * self.logit_scale.exp() + self.logit_bias

        idx = torch.arange(B, device=Z.device)
        if self.binary:
            T = torch.zeros_like(Z)
            T[idx, idx] = 1.0
        else:
            targets = labels.to(Z.dtype)
            T = torch.full_like(Z, 1.0 - self.easy_label)
            T[idx, idx] = 1.0
            T[idx, B + idx] = 1.0 - targets

        # Twin-positive hazard as in the other batch-wide losses: train.py always trains
        # this loss under NO_DUPLICATES.
        cell_loss = nn.functional.binary_cross_entropy_with_logits(Z, T, reduction="none")

        labeled_mask = torch.zeros_like(Z, dtype=torch.bool)
        labeled_mask[idx, idx] = True
        labeled_mask[idx, B + idx] = True
        hard_term = cell_loss[labeled_mask].mean()
        easy_cells = cell_loss[~labeled_mask]
        # Same B=1 tail-batch hazard as BatchGradedMarginMSELoss.
        if easy_cells.numel() == 0:
            return hard_term
        easy_term = easy_cells.mean()
        return self.hard_weight * hard_term + (1.0 - self.hard_weight) * easy_term
