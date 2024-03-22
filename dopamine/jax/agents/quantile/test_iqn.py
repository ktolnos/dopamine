import numpy as np
import jax.numpy as jnp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from flax import linen as nn
import jax
from dopamine.discrete_domains import atari_lib
from jax import random
import optax

kappa = 1
num_tau_samples = 64
num_tau_prime_samples = 64


data = sm.datasets.engel.load_pandas().data
data = data[data.income < 1500]
data = data[data.foodexp < 1000]
key1, key2 = random.split(random.key(0))
x = jnp.array(data.income)[:, None]
y = jnp.array(data.foodexp)[:, None]


class ImplicitQuantileNetwork(nn.Module):
    """The Implicit Quantile Network (Dabney et al., 2018).."""
    quantile_embedding_dim: int

    @nn.compact
    def __call__(self, x, num_quantiles, rng):
        initializer = nn.initializers.variance_scaling(
            scale=1.0 / jnp.sqrt(3.0),
            mode='fan_in',
            distribution='uniform')
        state_vector_length = x.shape[-1]
        state_net_tiled = jnp.tile(x, [num_quantiles, 1])
        quantiles_shape = [num_quantiles, 1]
        quantiles = jax.random.uniform(rng, shape=quantiles_shape)
        quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
        quantile_net = (
                jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32)
                * np.pi
                * quantile_net)
        quantile_net = jnp.cos(quantile_net)
        quantile_net = nn.Dense(features=state_vector_length,
                                kernel_init=initializer)(quantile_net)
        quantile_net = nn.relu(quantile_net)
        x = state_net_tiled * quantile_net
        x = nn.Dense(features=32, kernel_init=initializer)(x) #TODO 512
        x = nn.relu(x)
        n_actions = 1
        quantile_values = nn.Dense(features=n_actions,
                                   kernel_init=initializer)(x)
        return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)


# %%
network_def = ImplicitQuantileNetwork(quantile_embedding_dim=32)
batch_size = x.shape[0]

def loss_fn(params, rng_input, target_quantile_vals):
    def online(state, key):
        return network_def.apply(params, state, num_quantiles=num_tau_samples,
                                 rng=key)

    batched_rng = jnp.stack(jax.random.split(rng_input, num=batch_size))
    model_output = jax.vmap(online)(x, batched_rng)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    chosen_action_quantile_values = quantile_values #TODO
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_vals[:, :, None, :] -
                      chosen_action_quantile_values[:, None, :, :])
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = (
            (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
            0.5 * bellman_errors ** 2)
    huber_loss_case_two = (
            (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
            kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))
    huber_loss = huber_loss_case_one + huber_loss_case_two
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(quantiles[:, None, :, :],
                         [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))) * huber_loss) / kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = jnp.sum(quantile_huber_loss, axis=2)
    loss = jnp.mean(loss, axis=1)
    return jnp.mean(loss)


# %%
def train_quantile_model():
    key1, key2 = random.split(random.PRNGKey(0))

    params = network_def.init(key2, x[0], num_tau_samples, key1)  # Initialization call

    learning_rate = 0.3  # Gradient step size.
    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)
    y_quant = jnp.tile(y, [1, num_tau_prime_samples])[:, : , None]
    for i in range(101):
        key1, key2 = random.split(key1)
        loss_val, grads = loss_grad_fn(params, key2, y_quant)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            print('Loss step {}: '.format(i), loss_val)
    return params

# quant_params = train_quantile_model()
# x = jnp.arange(data.income.min(), data.income.max(), 50)[:, None]
# network_def.apply(quant_params, x, 10, key1)