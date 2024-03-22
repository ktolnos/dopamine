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


x_raw = np.random.uniform(0,1, size = (1000,))
y = x_raw * 2 + (x_raw+0.1) * np.random.normal(0, 0.3, size = (1000,))
n_sines = 3
sines = [jnp.sin(x_raw * i) for i in range(1, n_sines+1)]
cosines = [jnp.cos(x_raw * i) for i in range(1, n_sines+1)]
x = jnp.stack([x_raw, *sines, *cosines], axis=-1)


v_max = jnp.max(y)+1
v_min = jnp.min(y)-1
nr_bins = 51
bin_width = (v_max - v_min) / nr_bins
sigma_to_final_sigma_ratio = 0.75
support = jnp.linspace(v_min, v_max, nr_bins + 1, dtype=jnp.float32)
centers = (support[:-1] + support[1:]) / 2
sigma = bin_width * sigma_to_final_sigma_ratio

class ImplicitQuantileCrossentropyNetwork(nn.Module):
    """The Implicit Quantile Network (Dabney et al., 2018).."""
    quantile_embedding_dim: int

    @nn.compact
    def __call__(self, x, num_quantiles, rng):
        initializer = nn.initializers.variance_scaling(
            scale=1.0 / jnp.sqrt(3.0),
            mode='fan_in',
            distribution='uniform')
        x = nn.Dense(features=32, kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32, kernel_init=initializer)(x)
        x = nn.relu(x)
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
        x = nn.Dense(features=32, kernel_init=initializer)(x)
        x = nn.relu(x)
        n_actions = 1
        quantile_values = nn.Dense(features=n_actions * nr_bins,
                                   kernel_init=initializer)(x)
        quantile_values = quantile_values.reshape(-1, n_actions, nr_bins)
        return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)


# %%
network_crossent = ImplicitQuantileCrossentropyNetwork(quantile_embedding_dim=32)
batch_size = x_raw.shape[0]


def hl_gauss_encode(x):
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs /= z
    return target_probs


def convert_prob_to_value(probs):
    return jnp.sum(probs * centers, axis=-1)


# target_quantile_vals are probalilties for each bin with shape batch_size x num_tau_prime_samples x num_bins
def crossent_loss_fn(params, rng_input, target_quantile_vals):
    def online(state, key):
        return network_crossent.apply(params, state, num_quantiles=num_tau_samples,
                                 rng=key)
    # target_quantile_vals = jax.vmap(jax.vmap(hl_gauss_encode))(target_quantile_vals) TODO?

    batched_rng = jnp.stack(jax.random.split(rng_input, num=batch_size))
    model_output = jax.vmap(online)(x, batched_rng)
    quantile_values_logits = model_output.quantile_values
    quantiles = model_output.quantiles
    chosen_action_quantile_values_logits = quantile_values_logits[:, :, 0, :] #TODO later
    # Shape of bellman_erors and crossent_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.

    crossent_loss = optax.softmax_cross_entropy(
        chosen_action_quantile_values_logits[:, None, :, :],
        target_quantile_vals[:, :, None, :]
    )[:, :, :, None]

    target_quantile_vals_unbinned = jax.vmap(jax.vmap(convert_prob_to_value))(target_quantile_vals)
    chosen_action_quantile_values_unbinned = jax.vmap(jax.vmap(convert_prob_to_value))(
        jax.nn.softmax(chosen_action_quantile_values_logits, axis=-1))

    overestimation = (chosen_action_quantile_values_unbinned[:, None, :] > target_quantile_vals_unbinned[:, :, None]
                      )[:, :, :, None]
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(quantiles[:, None, :, :],
                         [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        overestimation.astype(jnp.float32))) * crossent_loss)
    # quantile_loss = crossent_loss
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = jnp.sum(quantile_loss, axis=2)
    loss = jnp.mean(loss, axis=1)
    return jnp.mean(loss)


# %%
def train_quantile_crossent_model():
    key1, key2 = random.split(random.PRNGKey(0))

    params = network_crossent.init(key2, x[0], num_tau_samples, key1)  # Initialization call

    learning_rate = 0.003  # Gradient step size.
    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(crossent_loss_fn)
    y_quant = jnp.tile(y[:, None], [1, num_tau_prime_samples])[:, :, None]
    y_encoded = jax.vmap(jax.vmap(hl_gauss_encode))(y_quant)  # 1, num_tau_prime_samples, batch, num_bins
    for i in range(101):
        key1, key2 = random.split(key1)
        loss_val, grads = loss_grad_fn(params, key2, y_encoded)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            print('Loss step {}: '.format(i), loss_val)
    return params


crossent_params = train_quantile_crossent_model()