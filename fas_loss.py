import torch
import torch.nn as nn

class FairAdaptiveScalingLoss(nn.Module):
    def __init__(self, num_groups: int, num_samples: int,
                 c: float = 0.5, ema_alpha: float = 0.1,
                 clip_min: float = 0.5, clip_max: float = 1.5):
        super().__init__()
        self.c = c
        self.ema_alpha = ema_alpha
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.register_buffer("l_ema", torch.zeros(num_samples))
        self.beta = nn.Embedding(num_groups, 1)
        nn.init.constant_(self.beta.weight, 1.0)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, group_ids, indices, enable_fas=True):
        ce_i = self.ce(logits, targets)
        with torch.no_grad():
            self.l_ema[indices] = (1 - self.ema_alpha) * self.l_ema[indices] + self.ema_alpha * ce_i.detach()
        if not enable_fas:
            return ce_i.mean(), torch.ones_like(ce_i)
        li = self.l_ema[indices]
        li_scaled = torch.sigmoid((li - li.mean()) / (li.std() + 1e-6))
        indiv = 0.5 + li_scaled
        beta = self.beta(group_ids).squeeze(-1)
        w = self.c * indiv + (1 - self.c) * beta
        w = w / (w.mean().detach() + 1e-6)
        w = torch.clamp(w, self.clip_min, self.clip_max)
        return (w * ce_i).mean(), w
