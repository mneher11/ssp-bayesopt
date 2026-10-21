import numpy as np
import nengo
from .vens import VirtualEnsemble


class MagnitudePrunedEncoders(nengo.dists.Distribution):
    """Sample default Nengo encoders, then globally prune smallest magnitudes."""

    sparsity = nengo.params.NumberParam("sparsity", low=0.0, high=1.0)

    def __init__(self, sparsity):
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("encoder sparsity must be in [0, 1]")
        super().__init__()
        self.sparsity = sparsity
        self.base_distribution = nengo.dists.ScatteredHypersphere(surface=True)

    def sample(self, n, d=None, rng=np.random):
        """Prune globally, then normalize surviving rows while retaining zeros."""
        samples = np.asarray(self.base_distribution.sample(n, d=d, rng=rng))
        if samples.ndim != 2:
            raise ValueError("encoder samples must be a 2D matrix")
        n_prune = int(round(self.sparsity * samples.size))
        if n_prune:
            prune_idx = np.argsort(np.abs(samples), axis=None, kind="stable")[:n_prune]
            samples.flat[prune_idx] = 0.0
        if self.sparsity == 0.0:
            return samples

        # Normalization is performed here (and disabled in the Ensemble) so
        # fully pruned rows can safely remain zero instead of becoming NaNs.
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        nonzero_rows = norms[:, 0] > 0.0
        samples[nonzero_rows] /= norms[nonzero_rows]
        return samples


# Notes regarding the parition:
# One chip has 128 cores, each core has
# MAX_NEURONS = 1024
# MAX_IN_AXONS = 4096
# MAX_OUT_AXONS = 4096
# Nengo will try to put one ensemble on one core and split if the number of neurons is too high
# but nengo will not split if the neuron of neurons is

def make_network(bo_soln_init, m, sigma, beta_inv, gamma_t,
                 neurons_per_dim=50, seed=0, tau=0.05,
                 var_weight=1.,
                 partition=None,
                 dt=0.001,
                 tau_probe=0.1,
                 rate=1.,
                 bias_free=False,
                 tau_ref=0.0,
                 neuron_type=nengo.LIF(),
                 decoder_solver=None,
                 encoder_sparsity=0.0):

    init_val = bo_soln_init.flatten()/np.linalg.norm(bo_soln_init)
    ssp_dim = bo_soln_init.size
    model = nengo.Network(seed=seed)

    # tau_ref: the YANA hardware has no refractory period, so the default 0
    # builds the hardware-faithful LIF (decoders solved against tau_ref=0
    # rate curves; solving them against tau_ref!=0 rates costs ~15x decode
    # error on hardware). None keeps the passed neuron_type untouched; a
    # numeric value rebuilds the neuron_type with that refractory period,
    # so "tau_ref=0.002" restores stock nengo.LIF behaviour.
    if tau_ref is not None:
        neuron_type = nengo.LIF(tau_rc=neuron_type.tau_rc, tau_ref=tau_ref)
    model.config[nengo.Ensemble].neuron_type = neuron_type

    n_neurons = neurons_per_dim * ssp_dim
    if not 0.0 <= encoder_sparsity <= 1.0:
        raise ValueError("encoder_sparsity must be in [0, 1]")
    encoder_kwargs = (
        {"encoders": MagnitudePrunedEncoders(encoder_sparsity)}
        if encoder_sparsity > 0.0 else {}
    )
    if encoder_sparsity > 0.0:
        # Sparse samples are normalized by MagnitudePrunedEncoders; leave rows
        # that prune to zero intact rather than dividing them by zero.
        model.config[nengo.Ensemble].normalize_encoders = False

    with model:
        def stim_func(t):
            if t < 0.1:
                return init_val
            else:
                return np.zeros((ssp_dim,))

        def transform_func(x, sigma=sigma, mu=m, beta_inv=beta_inv, gamma_t=gamma_t, var_weight=var_weight):
            sqr = np.dot(x, np.dot(sigma, x))
            scale = np.sqrt(sqr + gamma_t + beta_inv)
            a_out = (mu + var_weight * sigma @ x / scale)
            return tau * rate * a_out + x

        stim = nengo.Node(stim_func, label='stim')


        if partition is None:
            intercept_dist = nengo.dists.CosineSimilarity(ssp_dim + 2)
            if bias_free:
                # Legacy no-bias tuning (kept for A/B comparison against the
                # old backend): nengo-yana now supports biases natively (they
                # ride the export's per-chain Affine nodes), so the default
                # bias_free=False is the production path. Here gain =
                # 1/|alpha| with bias = 0 preserves each neuron's cosine
                # threshold, dropping only the negative-intercept
                # (spontaneously active) neurons. Floor |alpha| at 0.1 so
                # gains stay <= 10.
                mags = np.maximum(
                    np.abs(intercept_dist.sample(
                        n_neurons, rng=np.random.RandomState(seed))),
                    0.1)
                solution_neurons = nengo.Ensemble(
                    n_neurons, ssp_dim, label='solution_neurons',
                    **encoder_kwargs)
                solution_neurons.gain = 1.0 / mags
                solution_neurons.bias = np.zeros(n_neurons)
            else:
                solution_neurons = nengo.Ensemble(
                    n_neurons, ssp_dim, intercepts=intercept_dist,
                    label='solution_neurons', **encoder_kwargs)
            nengo.Connection(stim, solution_neurons, synapse=None)
            connection_kwargs = ({} if decoder_solver is None
                                 else {"solver": decoder_solver})
            nengo.Connection(solution_neurons, solution_neurons,
                             function=transform_func, synapse=tau,
                             **connection_kwargs)
            solution_probe = nengo.Probe(solution_neurons, synapse=tau_probe)

        else:
            if n_neurons % partition != 0:
                raise ValueError("n_neurons (%s) must be divisible by partition (%s)" % (
                    n_neurons, partition))

            solution_neurons = VirtualEnsemble(n_ensembles=partition,
                                n_neurons_per_ensemble=n_neurons // partition,
                                dimensions=ssp_dim, label='solution_neurons',
                                bias_free=bias_free, **encoder_kwargs)
            solution_neurons.add_input(stim, synapse=None)

            output_kwargs = ({} if decoder_solver is None
                             else {"solver": decoder_solver})
            solution_neurons.add_input(
                solution_neurons.add_output(
                    function=transform_func, dt=dt, **output_kwargs)[0],
                synapse=tau)

            # Copy the output so that the above is collapsed as a passthrough
            solution_probe = nengo.Probe(solution_neurons.add_output(dt=dt)[0], synapse=tau_probe)

    return model, solution_probe, stim

if __name__ == '__main__':

    ssp_dim = 15

    sigma = np.eye(ssp_dim)
    mu = np.random.random(size=(ssp_dim,))
    step_size = 0.1
    beta_inv = 2
    gamma_t = 0
    init_guess = 2*(np.random.random(size=(ssp_dim,)) - 0.5 )
    # init_guess = np.ones((ssp_dim,))
    # init_guess = np.random.random(size=(ssp_dim,))

    def _pick_simulator():
        """Pick a simulator backend for the toy problem.

        Preference order: nengo_loihi > reference nengo.
        """
        try:
            import nengo_loihi
            return nengo_loihi.Simulator
        except ImportError:
            pass
        return nengo.Simulator

    Sim = _pick_simulator()
    model, solution_probe, _ = make_network(
            bo_soln_init=init_guess,
            m=mu,
            sigma=sigma,
            beta_inv=beta_inv,
            gamma_t=gamma_t,
            neurons_per_dim=4,
            partition=None,
        )
    sim = Sim(model)
    with sim:
        sim.run(2.5)
    raw_data = sim.data[solution_probe]
    print("Done no partition")

    model, solution_probe, _ = make_network(
        bo_soln_init=init_guess,
        m=mu,
        sigma=sigma,
        beta_inv=beta_inv,
        gamma_t=gamma_t,
        neurons_per_dim=8,
        partition=1,
    )
    sim = Sim(model)
    with sim:
        sim.run(2.5)
    raw_data_v2 = sim.data[solution_probe]
    print("Done with partition")

    def get_fun(data):
        """Acquisition value a(φ) = m·φ + sqrt(β⁻¹ + φᵀΣφ) along a trajectory."""
        return data @ mu + np.sqrt(beta_inv + np.sum((data @ sigma) * data, axis=-1))

    t = sim.trange()

    def norm(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    n_raw = norm(raw_data)
    n_raw_v2 = norm(raw_data_v2)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Panel 1: normalized solution-SSP trajectory, one line per dimension
    # (ssp_dim lines per run). The network performs gradient ascent on the
    # acquisition in SSP space; convergence shows as all dimensions of a run
    # settling to the same fixed point.
    axs[0].plot(t, n_raw, color='tab:blue', alpha=0.8, lw=0.8)
    axs[0].plot(t, n_raw_v2, '--', color='tab:red', alpha=0.8, lw=0.8)
    axs[0].set_title('Solution-SSP trajectory (normalized)')
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel(r'$\phi_t / \|\phi_t\|$')

    # Panel 2: acquisition value a(φ) = m·φ + √(β⁻¹ + φᵀΣφ) along each
    # normalized trajectory. The recurrent connection implements one
    # gradient-ascent step per timestep; a successful solve makes both
    # curves rise and plateau at the same value.
    axs[1].plot(t, get_fun(n_raw), color='tab:blue', alpha=0.8, lw=1.2)
    axs[1].plot(t, get_fun(n_raw_v2), '--', color='tab:red', alpha=0.8, lw=1.2)
    axs[1].set_title('Acquisition value along trajectory')
    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel(r'$m\cdot\phi + \sqrt{\beta^{-1} + \phi^T\Sigma\phi}$')

    # One figure-level legend with proxy handles: the panel-1 plots draw
    # ssp_dim lines per call, so per-axes legends would list every dimension.
    handles = [
        Line2D([], [], color='tab:blue', lw=2,
               label='single Ensemble (partition=None, 4 neurons/dim)'),
        Line2D([], [], color='tab:red', ls='--', lw=2,
               label='VirtualEnsemble (partition=1, 8 neurons/dim)'),
    ]
    fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False)

    fig.savefig('network_solver_demo.png', dpi=150)
    plt.show()
