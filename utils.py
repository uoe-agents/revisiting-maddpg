import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import jax.lax as jlax
import equinox as eqx

def gumbel_softmax(logits, key, temperature=1.0, st=True):
    # TODO: Explore how this could be written with a custom jax derivative expression (jax.custom_jvp)

    gumbels = jrand.gumbel(key, logits.shape)
    y_soft = jnn.softmax( (logits + gumbels) / temperature )
    
    if st:
        y_hard = jnn.one_hot(jnp.argmax(y_soft), num_classes=y_soft.shape[-1])
        # See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
        return y_hard - jlax.stop_gradient(y_soft) + y_soft
    else:
        return y_soft

def _incremental_update(old, new, tau):
    if new is None:
        return old
    else:
        return (1 - tau) * old + tau * new

def _is_none(x):
    return x is None

def soft_update(target_model, behaviour_model, tau):
    return jax.tree_map(
        lambda old, new : _incremental_update(old, new, tau),
        target_model,
        eqx.filter(behaviour_model, eqx.is_array),
        is_leaf=_is_none,
    )