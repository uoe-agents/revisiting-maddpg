import equinox as eqx
import jax
import utils
import jax.random as jrand

class ActorNetwork(eqx.Module):
    layers: list
    #bias: jax.numpy.ndarray

    def __init__(self, obs_dim, n_actions, fc_dims, lr, key):
        # TODO: make this a generic n-layer network?
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(obs_dim, fc_dims[0], key=key1),
                       eqx.nn.Linear(fc_dims[0], fc_dims[1], key=key2),
                       eqx.nn.Linear(fc_dims[1], n_actions, key=key3)]

    def __call__(self, xx):
        for layer in self.layers[:-1]:
            xx = jax.nn.relu(layer(xx))
        #if self.is_discrete:
        #return utils.gumbel_softmax(self.layers[-1](xx), jrand.PRNGKey(123))
        #else: # TODO
        return jax.nn.softmax(self.layers[-1](xx))#, axis=1)

class CriticNetwork(eqx.Module):
    layers: list
    #bias: jax.numpy.ndarray

    def __init__(self, obs_dims, n_agents, n_actions, fc_dims, lr, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(obs_dims + n_agents*n_actions, fc_dims[0], key=key1),
                       eqx.nn.Linear(fc_dims[0], fc_dims[1], key=key2),
                       eqx.nn.Linear(fc_dims[1], 1, key=key3)]
        # TODO : Add bias??

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)# + self.bias

# @jax.jit
# @jax.grad
# def loss(model, x, y):
#     pred_y = jax.vmap(model)(x)
#     return jax.numpy.mean((y - pred_y) ** 2)

# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# x = jax.random.normal(x_key, (100, 2))
# y = jax.random.normal(y_key, (100, 2))
# model = MyModule(model_key)
# grads = loss(model, x, y)
# learning_rate = 0.1
# model = jax.tree_map(lambda m, g: m - learning_rate * g, model, grads)