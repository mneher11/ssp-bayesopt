import numpy as np

import nengo

from ssp_bayes_opt.network_solver import MagnitudePrunedEncoders


def test_magnitude_pruned_encoders_prune_exact_global_fraction():
    n_neurons, dimensions, sparsity = 20, 97, 0.8
    seed = 31
    base = nengo.dists.ScatteredHypersphere(surface=True).sample(
        n_neurons, dimensions, rng=np.random.RandomState(seed)
    )
    actual = MagnitudePrunedEncoders(sparsity).sample(
        n_neurons, dimensions, rng=np.random.RandomState(seed)
    )

    expected_pruned = round(sparsity * base.size)
    expected = base.copy()
    prune_idx = np.argsort(np.abs(expected), axis=None, kind="stable")[:expected_pruned]
    expected.flat[prune_idx] = 0.0
    expected_norms = np.linalg.norm(expected, axis=1, keepdims=True)
    expected_norms[expected_norms == 0.0] = 1.0
    expected /= expected_norms

    assert np.count_nonzero(actual == 0) == expected_pruned
    assert np.allclose(actual, expected)


def test_magnitude_pruned_encoders_preserve_default_samples_at_zero():
    n_neurons, dimensions, seed = 10, 97, 42
    expected = nengo.dists.ScatteredHypersphere(surface=True).sample(
        n_neurons, dimensions, rng=np.random.RandomState(seed)
    )
    actual = MagnitudePrunedEncoders(0.0).sample(
        n_neurons, dimensions, rng=np.random.RandomState(seed)
    )

    assert np.array_equal(actual, expected)


def test_high_sparsity_preserves_fully_pruned_encoders_as_zero_rows():
    samples = MagnitudePrunedEncoders(0.99).sample(
        20, 5, rng=np.random.RandomState(0)
    )

    row_norms = np.linalg.norm(samples, axis=1)
    assert np.any(row_norms == 0.0)
    assert np.all(np.isclose(row_norms, 0.0) | np.isclose(row_norms, 1.0))
    assert not np.any(np.isnan(samples))


def test_fully_pruned_encoder_distribution_returns_all_zero_matrix():
    samples = MagnitudePrunedEncoders(1.0).sample(
        20, 5, rng=np.random.RandomState(0)
    )

    assert np.count_nonzero(samples) == 0
