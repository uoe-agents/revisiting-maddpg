import torch
from torch import Tensor
from torch.distributions import Gumbel, Exponential, OneHotCategorical
from torch.nn.functional import one_hot, softmax

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

class GradientEstimator: # TODO : Actually leverage parent class
    pass

class STGS(GradientEstimator):
    """
        Straight-Through Gumbel Softmax estimator
    """
    def __init__(self, temperature):
        self.temperature = temperature
        self.gumbel_dist = Gumbel(loc=Tensor([0]), scale=Tensor([1]))

    def __call__(self, logits):
        gumbel_noise = self.gumbel_dist.sample(logits.shape).squeeze(-1) # ~ Gumbel (0,1)
        perturbed_logits = (logits + gumbel_noise) / self.temperature  # ~ Gumbel(logits,tau)
        y_soft = softmax(perturbed_logits, dim=-1)
        y_hard = one_hot(y_soft.argmax(dim=-1), num_classes=logits.shape[-1])
        return replace_gradient(value=y_hard, surrogate=y_soft)

class GRMCK(GradientEstimator):
    """
        Gumbel-Rao Monte-Carlo estimator

        Credit: https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py
    """
    def __init__(self, temperature, kk):
        self.temperature = temperature
        self.kk = kk
        self.exp_dist = Exponential(rate=Tensor([1]))

    @torch.no_grad()
    def _conditional_gumbel(self, logits, DD):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one hot vector)."""
        # iid. exponential
        EE = self.exp_dist.sample([self.kk, *logits.shape]).squeeze(-1)
        # E of the chosen class
        Ei = (DD * EE).sum(dim=-1, keepdim=True)
        # partition function (normalization constant)
        ZZ = logits.exp().sum(dim=-1, keepdim=True)
        # Sampled gumbel-adjusted logits
        adjusted = (DD * (-torch.log(Ei) + torch.log(ZZ)) +
                   (1 - DD) * -torch.log(EE/torch.exp(logits) + Ei / ZZ))
        return adjusted - logits

    def __call__(self, logits):
        DD = OneHotCategorical(logits=logits).sample()
        adjusted = logits + self._conditional_gumbel(logits, DD)
        surrogate = softmax(adjusted/self.temperature, dim=-1).mean(dim=0)
        return replace_gradient(DD, surrogate)

class GST(GradientEstimator):
    """
        Gapped Straight-Through Estimator

        Credit: https://github.com/chijames/GST/blob/267ab3aa202d7a0cfd5b5861bd3dcad87faefd9f/model/basic.py
    """
    def __init__(self, temperature, gap):
        self.temperature = temperature
        self.gap = gap

    def __call__(self, logits):
        logits_cpy = logits.detach()
        probs = torch.nn.functional.softmax(logits_cpy, dim=-1)
        mm = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
        action = mm.sample() 
        argmax = probs.argmax(dim=-1, keepdim=True)
        
        action_bool = action.bool()
        max_logits = torch.gather(logits_cpy, -1, argmax)
        move = (max_logits - logits_cpy)*action

        move2 = ( logits_cpy + (-max_logits + self.gap) ).clamp(min=0.0)
        move2[action_bool] = 0.0 # Equivalent to .(1-D)
        logits = logits + (move - move2)

        logits = logits - logits.mean(dim=-1, keepdim=True)
        prob = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        action = action - prob.detach() + prob
        return action.reshape(logits.shape)
