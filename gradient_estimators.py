import numpy as np
import torch
from torch import Tensor
from torch.distributions import Gumbel, Exponential, OneHotCategorical
from torch.nn.functional import one_hot, softmax

GUMBEL_DIST = Gumbel(loc=Tensor([0]), scale=Tensor([1]))
EXP_DIST = Exponential(rate=Tensor([1]))

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

class GradientEstimator:
    def update_state(self):
        ...

class STGS(GradientEstimator):
    """
        Straight-Through Gumbel Softmax estimator
    """
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, logits, need_gradients=None):
        gumbel_noise = GUMBEL_DIST.sample(logits.shape).squeeze(-1) # ~ Gumbel (0,1)
        perturbed_logits = (logits + gumbel_noise) / self.temperature  # ~ Gumbel(logits,tau)
        y_soft = softmax(perturbed_logits, dim=-1)
        y_hard = one_hot(y_soft.argmax(dim=-1), num_classes=logits.shape[-1])
        return replace_gradient(value=y_hard, surrogate=y_soft)

class TAGS(STGS):
    """
        Temperature-Annealed Straight-Through Gumbel Softmax estimator
        Annealing scheme: Decaying exponential
    """
    def __init__(self, start_temp, end_temp, period):
        super().__init__(start_temp)
        self.multiplier = (end_temp / start_temp) ** (1 / period)
        self.period = period
        self.updates = 0

    def update_state(self):
        if (self.updates < self.period):
            self.temperature *= self.multiplier
        self.updates += 1

class GRMCK(GradientEstimator):
    """
        Gumbel-Rao Monte-Carlo estimator

        With help from: https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py
    """
    def __init__(self, temperature, kk):
        self.temperature = temperature
        self.kk = kk

    @torch.no_grad()
    def _conditional_gumbel(self, logits, DD):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one hot vector)."""
        # iid. exponential
        EE = EXP_DIST.sample([self.kk, *logits.shape]).squeeze(-1)
        # E of the chosen class
        Ei = (DD * EE).sum(dim=-1, keepdim=True)
        # partition function (normalization constant)
        ZZ = logits.exp().sum(dim=-1, keepdim=True)
        # Sampled gumbel-adjusted logits
        adjusted = (DD * (-torch.log(Ei) + torch.log(ZZ)) +
                   (1 - DD) * -torch.log(EE/torch.exp(logits) + Ei / ZZ))
        return adjusted - logits

    def __call__(self, logits, need_gradients=True):
        DD = OneHotCategorical(logits=logits).sample()
        if need_gradients:
            adjusted = logits + self._conditional_gumbel(logits, DD)
            surrogate = softmax(adjusted/self.temperature, dim=-1).mean(dim=0)
            return replace_gradient(DD, surrogate)
        else:
            return DD

class GST(GradientEstimator):
    """
        Gapped Straight-Through Estimator

        With help from: https://github.com/chijames/GST/blob/267ab3aa202d7a0cfd5b5861bd3dcad87faefd9f/model/basic.py
    """
    def __init__(self, temperature, gap=1.0):
        self.temperature = temperature
        self.gap = gap

    @torch.no_grad()
    def _calculate_movements(self, logits, DD):
        max_logit = logits.max(dim=-1, keepdim=True)[0]
        selected_logit = torch.gather(logits, dim=-1, index=DD.argmax(dim=-1, keepdim=True))
        m1 = (max_logit - selected_logit) * DD
        m2 = (logits + self.gap - max_logit).clamp(min=0.0) * (1 - DD)
        return m1, m2

    def __call__(self, logits, need_gradients=True):
        DD = OneHotCategorical(logits=logits).sample()
        if need_gradients:
            m1, m2 = self._calculate_movements(logits, DD)
            surrogate = softmax((logits + m1 - m2) / self.temperature, dim=-1)
            return replace_gradient(DD, surrogate)
        else:
            return DD
