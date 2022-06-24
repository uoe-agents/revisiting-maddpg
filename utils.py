from jax import jit
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import jax.lax as jlax
import haiku as hk
from typing import Optional, Any

Array = Any

# TODO: Decide whethere 'hard' switch is necessary. By removing control flow, we can jit
#@jit
def gumbel_softmax_st(logits: Array, key, tau: float = 1) -> Array:
    """
        Straight-through Gumbel-Softmax Estimator
        See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
    """
    gumbels = jrand.gumbel(key, logits.shape)
    y_soft = jnn.softmax( (logits + gumbels) / tau )

    y_hard = jnn.one_hot(jnp.argmax(y_soft), num_classes=logits.shape[-1])
    return y_hard - jlax.stop_gradient(y_soft) + y_soft

def _hk_tt(xx):
    return hk.without_apply_rng(hk.transform(xx))
