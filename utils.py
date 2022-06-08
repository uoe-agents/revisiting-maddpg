import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import jax.lax as jlax

def gumbel_softmax(logits, key, temperature=1.0, st=True):
    gumbels = jrand.gumbel(key, logits.shape)
    y_soft = jnn.softmax( (logits + gumbels) / temperature )
    
    if st:
        y_hard = jnn.one_hot(jnp.argmax(y_soft), num_classes=y_soft.shape[-1])
        # See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
        return y_hard - jlax.stop_gradient(y_soft) + y_soft
    else:
        return y_soft