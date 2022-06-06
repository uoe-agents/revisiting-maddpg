import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn

key = jrand.PRNGKey(0)

def gumbel_noise(shape, key, eps=1e-20):
    uu = jrand.uniform(key, shape)
    return -jnp.log( -jnp.log( uu + eps) + eps)

def gumbel_softmax(logits, key, temperature=1.0, st=True):
    yy = logits + gumbel_noise(logits.shape, key)
    if st:
        return jnn.one_hot(jnp.argmax(yy), yy.shape[0])
    else:
        return jnn.softmax(yy / temperature) #, dim=1) #TODO: Sort out batching

