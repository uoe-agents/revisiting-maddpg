import haiku as hk
import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap

class SimpleNetwork(hk.Module):
    def __call__(self, xx: jnp.ndarray) -> jnp.DeviceArray:
        net = hk.Sequential(layers=[hk.Linear(3)])
        return net(xx) + jrand.normal(hk.next_rng_key(), (3,))

rng = hk.PRNGSequence(0)
key = jrand.PRNGKey(0)
dummy_input = jnp.ones((1,3))

# ITERATION INF
# network = hk.transform(hk.vmap(lambda xx : SimpleNetwork()(xx), out_axes=(0,1), split_rng=True))
# params = network.init(key, dummy_input)
# dummy_output = network.apply(params, key, dummy_input)

# print(dummy_output)


# ITERATION 1: Simple, works
network = hk.transform(lambda xx : SimpleNetwork()(xx))
params = network.init(key, dummy_input)
dummy_output = network.apply(params, key, dummy_input)

print(dummy_output)

# ITERATION 2: vmap --> Problem! Doesn't advance the RNG state across the vmap
BATCH_SIZE = 8
batched_dummy_input = jnp.ones((BATCH_SIZE, INPUT_DIM))
batched_dummy_output = vmap(network.apply, in_axes=(None,None,0))(params, key, batched_dummy_input)

print(batched_dummy_output)

# # ITERATION 3
batched_network = hk.vmap(network, split_rng=True) # Doesn't work -- batched_network is just a function now... can't call init

# batched_network = hk.vmap(network.apply, split_rng=True)
# batched_dummy_output = batched_network(params, key, batched_dummy_input)

# print(batched_dummy_output)