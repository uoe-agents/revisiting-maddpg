import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import jax.lax as jlax
import haiku as hk
from typing import Optional

def gumbel_softmax(logits, key, temperature, st=True):
    # TODO: Explore how this could be written with a custom jax derivative expression (jax.custom_jvp)

    gumbels = jrand.gumbel(key, logits.shape)
    y_soft = jnn.softmax( (logits + gumbels) / temperature )
    
    if st:
        y_hard = jnn.one_hot(jnp.argmax(y_soft), num_classes=y_soft.shape[-1])
        # See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
        return y_hard - jlax.stop_gradient(y_soft) + y_soft
    else:
        return y_soft

class GumbelSoftmax(hk.Module):
    """TODO"""

    def __init__(
        self,
        temperature: float,
        key,
        straight_through: Optional[bool] = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.temperature = temperature
        self.key = key
        self.st = straight_through

    def __call__(
        self,
        logits: jnp.ndarray,
        *args,
    ) -> jnp.ndarray:    
        self.key, subkey = jrand.split(self.key)
        gumbels = jrand.gumbel(subkey, logits.shape)
        y_soft = jnn.softmax( (logits + gumbels) / self.temperature )
    
        if self.st:
            y_hard = jnn.one_hot(jnp.argmax(y_soft), num_classes=logits.shape[-1])
            # See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
            return y_hard - jlax.stop_gradient(y_soft) + y_soft
        else:
            return y_soft

def _hk_tt(xx):
    return hk.without_apply_rng(hk.transform(xx))
