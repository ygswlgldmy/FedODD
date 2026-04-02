import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.utils import get_root_logger
from functools import partial

from ..builder import LOSSES


@LOSSES.register_module()
class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        logger = get_root_logger()
        logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


@LOSSES.register_module()
class EQLv2Enhanced(nn.Module):
    """EQL v2 with dual-EMA gradient tracking, rare-class prior, and trend term.

    Changes vs. EQL v2:
      - Cumulative gradient ratio replaced by fast/slow EMA gradient ratio.
      - Positive-weight formula gains a rare-class prior c_j and trend term d_{j,t}.
      - Negative-weight formula unchanged (driven solely by fast gradient ratio).

    Args:
        num_classes (int): Number of foreground classes (excluding background).
        gamma (float): Slope of the sigmoid mapping function.
        mu (float): Centre of the sigmoid mapping function.
        alpha (float): Base positive-enhancement coefficient.
        fast_beta (float): EMA decay for fast (short-term) statistics. Closer
            to 0 → more reactive. Default 0.9.
        slow_beta (float): EMA decay for slow (long-term) statistics. Default 0.99.
        rho (float): Strength of the rare-class prior c_j. Default 0.5.
        lambda_trend (float): Weight applied to the trend term d_{j,t}. Default 1.0.
        num_samples_per_cls (list[int] | None): Per-class sample counts used to
            compute c_j. If None, all classes are treated as equally frequent
            (c_j = 1 for all j).
    """

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 fast_beta=0.9,
                 slow_beta=0.99,
                 rho=0.5,
                 lambda_trend=1.0,
                 num_samples_per_cls=None,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.fast_beta = fast_beta
        self.slow_beta = slow_beta
        self.rho = rho
        self.lambda_trend = lambda_trend
        self.test_with_obj = test_with_obj

        # ---- mapping function (same as EQL v2) ----
        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

        # ---- fast EMA gradient buffers ----
        # Initialised so that pos/neg ratio ≈ 100, matching EQL v2's convention
        # (f(100) ≈ 1 → no suppression / no boost at the start of training).
        self.register_buffer('pos_grad_fast', torch.ones(num_classes) * 100)
        self.register_buffer('neg_grad_fast', torch.ones(num_classes))
        self.register_buffer('pos_neg_fast',  torch.ones(num_classes) * 100)

        # ---- slow EMA gradient buffers ----
        self.register_buffer('pos_grad_slow', torch.ones(num_classes) * 100)
        self.register_buffer('neg_grad_slow', torch.ones(num_classes))
        self.register_buffer('pos_neg_slow',  torch.ones(num_classes) * 100)

        # ---- rare-class prior c_j ----
        # c_j = 1 + rho * (log n_max - log n_j) / (log n_max - log n_min + eps)
        # Shape: [num_classes]
        if num_samples_per_cls is not None:
            counts = torch.tensor(num_samples_per_cls, dtype=torch.float32)
            assert counts.numel() == num_classes, (
                f"num_samples_per_cls length ({counts.numel()}) must equal "
                f"num_classes ({num_classes})")
            log_counts = torch.log(counts.clamp(min=1.0))
            log_max = log_counts.max()
            log_min = log_counts.min()
            c_prior = 1.0 + rho * (log_max - log_counts) / (log_max - log_min + 1e-10)
        else:
            c_prior = torch.ones(num_classes)
        self.register_buffer('c_prior', c_prior)

        logger = get_root_logger()
        logger.info(
            f"build EQLv2Enhanced — gamma:{gamma}, mu:{mu}, alpha:{alpha}, "
            f"fast_beta:{fast_beta}, slow_beta:{slow_beta}, "
            f"rho:{rho}, lambda_trend:{lambda_trend}, "
            f"num_samples_per_cls={'provided' if num_samples_per_cls else 'None'}")

    # ------------------------------------------------------------------
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    # ------------------------------------------------------------------
    def get_channel_num(self, num_classes):
        return num_classes + 1

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    # ------------------------------------------------------------------
    def collect_grad(self, cls_score, target, weight):
        """Update fast/slow EMA gradient statistics (foreground classes only)."""
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # Exclude objectiveness (background) channel [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        # Fast EMA update
        self.pos_grad_fast = self.fast_beta * self.pos_grad_fast + (1 - self.fast_beta) * pos_grad
        self.neg_grad_fast = self.fast_beta * self.neg_grad_fast + (1 - self.fast_beta) * neg_grad
        self.pos_neg_fast  = self.pos_grad_fast / (self.neg_grad_fast + 1e-10)

        # Slow EMA update
        self.pos_grad_slow = self.slow_beta * self.pos_grad_slow + (1 - self.slow_beta) * pos_grad
        self.neg_grad_slow = self.slow_beta * self.neg_grad_slow + (1 - self.slow_beta) * neg_grad
        self.pos_neg_slow  = self.pos_grad_slow / (self.neg_grad_slow + 1e-10)

    # ------------------------------------------------------------------
    def get_weight(self, cls_score):
        """Compute per-class positive and negative sample weights.

        r_{j,t} = f(g^f_{j,t})
        q_{j,t} = 1 + alpha * c_j * (1 + lambda_trend * d_{j,t}) * (1 - f(g^f_{j,t}))

        where
            d_{j,t} = ReLU( (g^s - g^f) / (g^s + eps) )
        """
        g_fast = self.pos_neg_fast   # [num_classes]
        g_slow = self.pos_neg_slow   # [num_classes]

        # Trend term: large when recent state is worse than long-term average
        d = F.relu((g_slow - g_fast) / (g_slow + 1e-10))  # [num_classes]

        f_fast = self.map_func(g_fast)  # [num_classes]

        # Negative suppression weight (background class = 1, no change)
        neg_w = torch.cat([f_fast, cls_score.new_ones(1)])  # [num_classes + 1]

        # Positive enhancement weight
        pos_w_fg = 1.0 + self.alpha * self.c_prior * (1.0 + self.lambda_trend * d) * (1.0 - f_fast)
        pos_w = torch.cat([pos_w_fg, cls_score.new_ones(1)])  # [num_classes + 1]

        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w
