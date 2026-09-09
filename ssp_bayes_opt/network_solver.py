import numpy as np
import nengo
from .vens import VirtualEnsemble

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
                 neuron_type=nengo.LIF()):

    init_val = bo_soln_init.flatten()/np.linalg.norm(bo_soln_init)
    ssp_dim = bo_soln_init.size
    model = nengo.Network(seed=seed)
    model.config[nengo.Ensemble].neuron_type = neuron_type

    n_neurons = neurons_per_dim * ssp_dim

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
            solution_neurons = nengo.Ensemble(n_neurons, ssp_dim, intercepts=nengo.dists.CosineSimilarity(ssp_dim+2),
                                              label='solution_neurons')
            nengo.Connection(stim, solution_neurons, synapse=None)
            nengo.Connection(solution_neurons, solution_neurons, function=transform_func,
                             synapse=tau)  # , solver=nengo.solvers.LstsqDrop(weights=False,drop=0.25))
            solution_probe = nengo.Probe(solution_neurons, synapse=tau_probe)

        else:
            if n_neurons % partition != 0:
                raise ValueError("n_neurons (%s) must be divisible by partition (%s)" % (
                    n_neurons, partition))

            solution_neurons = VirtualEnsemble(n_ensembles=partition,
                                n_neurons_per_ensemble=n_neurons // partition,
                                dimensions=ssp_dim, label='solution_neurons')
            solution_neurons.add_input(stim, synapse=None)

            solution_neurons.add_input(solution_neurons.add_output(function=transform_func, dt=dt)[0],
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
