import torch

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

# class GradientEstimator: # TODO: Make a parent class?

class STGS:
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


class GRMCK:
    """
        Gumbel-Rao Monte-Carlo estimator
    """
    def __init__(self, temperature, kk):
        self.temperature = temperature
        self.kk = kk
    
    @torch.no_grad()
    @staticmethod
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
