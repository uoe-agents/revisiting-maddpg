import torch

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
    
    def __call__(self, logits):
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.temperature  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(-1)

        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        return replace_gradient(value=y_hard, surrogate=y_soft)


class GRMCK(GradientEstimator):
    """
        Gumbel-Rao Monte-Carlo estimator

        Credit: https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py
    """
    def __init__(self, temperature, kk):
        self.temperature = temperature
        self.kk = kk

    def _conditional_gumbel(self, logits, DD):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one hot vector)."""
        # iid. exponential
        EE = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([self.kk])
        # E of the chosen class
        Ei = (DD * EE).sum(dim=-1, keepdim=True)
        # partition function (normalization constant)
        ZZ = logits.exp().sum(dim=-1, keepdim=True)
        # Sampled gumbel-adjusted logits
        adjusted = (DD * (-torch.log(Ei) + torch.log(ZZ)) +
                    (1 - DD) * -torch.log(EE/torch.exp(logits) + Ei / ZZ))
        return adjusted - logits

    def __call__(self, logits):
        num_classes = logits.shape[-1]
        II = torch.distributions.categorical.Categorical(logits=logits).sample()
        DD = torch.nn.functional.one_hot(II, num_classes).float()
        adjusted = logits + self._conditional_gumbel(logits, DD)
        surrogate = torch.nn.functional.softmax(adjusted/self.temperature, dim=-1).mean(dim=0)
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
