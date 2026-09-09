"""Head-to-head toy problems: reference nengo.Simulator vs nengo_yana.Simulator.

Builds the network_solver acquisition-ascent network for two toy posteriors
and runs each on both backends, comparing the probe trajectories.

The network is built exactly as the production BO path builds it (default
nengo.LIF() neurons, CosineSimilarity intercepts, non-zero bias). No
YANA-specific accommodations are made: if the YANA backend cannot yet
compile the network, the failure is reported and the reference result is
still shown.

Usage:
    python experiments/yana_toy_problem.py [--plot]
"""
import argparse
import os
import sys

import numpy as np
import nengo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssp_bayes_opt.network_solver import make_network  # noqa: E402


def make_case(name, ssp_dim, seed):
    """Toy posterior with a known, checkable structure.

    'random'   -- random posterior mean, identity Sigma. Same setup as
                  network_solver.__main__, so results are comparable.

    'rank-one' -- Sigma = I + m m^T with m a random unit vector. The
                  acquisition a(phi) = m.phi + sqrt(beta_inv + phi^T Sigma phi)
                  is maximized over unit phi in the direction of m, so the
                  correct answer is phi* = m and a(phi*) is computable in
                  closed form.
    """
    rng = np.random.default_rng(seed)
    if name == "random":
        m = rng.random(size=(ssp_dim,))
        sigma = np.eye(ssp_dim)
    elif name == "rank-one":
        m = rng.standard_normal(size=(ssp_dim,))
        m /= np.linalg.norm(m)
        sigma = np.eye(ssp_dim) + np.outer(m, m)
    else:
        raise ValueError(name)
    beta_inv, gamma_t = 2.0, 0.0
    init_guess = 2 * (rng.random(size=(ssp_dim,)) - 0.5)
    return dict(name=name, m=m, sigma=sigma, beta_inv=beta_inv,
                gamma_t=gamma_t, init_guess=init_guess, ssp_dim=ssp_dim)


def acquisition(case, phi):
    """Acquisition a(phi) = m.phi + sqrt(beta_inv + phi^T Sigma phi)."""
    m, sigma = case["m"], case["sigma"]
    phi = np.atleast_2d(phi)
    return phi @ m + np.sqrt(case["beta_inv"] + np.einsum("ij,jk,ik->i", phi, sigma, phi))


def run_case(case, Sim, sim_time=2.5, **net_kwargs):
    """Run one toy case on one backend; return (trajectory, solution, final acq)."""
    model, probe, _ = make_network(
        bo_soln_init=case["init_guess"],
        m=case["m"],
        sigma=case["sigma"],
        beta_inv=case["beta_inv"],
        gamma_t=case["gamma_t"],
        **net_kwargs,
    )
    sim = Sim(model, progress_bar=False)
    with sim:
        sim.run(sim_time)
    traj = sim.data[probe]
    soln = traj[-500:].mean(0)
    soln_unit = soln / (np.linalg.norm(soln) + 1e-12)
    return traj, soln, soln_unit


def normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n


def compare(traj_a, traj_b):
    """Trajectory-level agreement between two backends."""
    n = min(len(traj_a), len(traj_b))
    ta, tb = normalize(traj_a[:n]), normalize(traj_b[:n])
    diff = np.linalg.norm(ta - tb, axis=-1)
    return dict(rms=float(np.sqrt(np.mean(diff ** 2))), max=float(diff.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-time", type=float, default=2.5)
    parser.add_argument("--neurons-per-dim", type=int, default=4)
    parser.add_argument("--partition", type=int, default=None,
                        help="partition arg for make_network (VirtualEnsemble); default single ensemble")
    parser.add_argument("--ssp-dim", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="make_network seed")
    parser.add_argument("--plot", action="store_true", help="save comparison figure")
    args = parser.parse_args()

    net_kwargs = dict(neurons_per_dim=args.neurons_per_dim,
                      partition=args.partition,
                      seed=args.seed)

    # Backend registry: each entry is (label, Simulator class).
    backends = [("nengo", nengo.Simulator)]
    try:
        import nengo_yana
        backends.append(("nengo_yana", nengo_yana.Simulator))
    except ImportError:
        print("nengo_yana not importable -- running reference backend only.")

    for case_name in ("random", "rank-one"):
        case = make_case(case_name, args.ssp_dim, seed=42)
        init_n = normalize(case["init_guess"].reshape(1, -1))
        print(f"\n=== case '{case_name}' (ssp_dim={args.ssp_dim}) ===")
        print(f"    init acq  = {acquisition(case, init_n)[0]:.4f}")
        if case_name == "rank-one":
            print(f"    optimal acq (phi* = m) = {acquisition(case, case['m'].reshape(1, -1))[0]:.4f}")

        results = {}
        for label, Sim in backends:
            try:
                traj, soln, soln_unit = run_case(
                    case, Sim, sim_time=args.sim_time, **net_kwargs)
                acq_final = acquisition(case, soln_unit.reshape(1, -1))[0]
                results[label] = dict(traj=traj, soln=soln, soln_unit=soln_unit,
                                      acq=acq_final)
                print(f"    {label:12s} final acq = {acq_final:.4f}")
            except Exception as exc:
                import traceback
                tb = traceback.extract_tb(exc.__traceback__)
                frame = tb[-1]
                print(f"    {label:12s} FAILED: {type(exc).__name__} at "
                      f"{os.path.basename(frame.filename)}:{frame.lineno}: {exc}")

        if len(results) == 2:
            (la, ra), (lb, rb) = list(results.items())
            c = compare(ra["traj"], rb["traj"])
            cos = float(ra["soln_unit"] @ rb["soln_unit"])
            print(f"    agreement: rms={c['rms']:.4f}  max={c['max']:.4f}  cos(soln)={cos:.4f}")

        if args.plot and results:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            colors = {"nengo": "tab:blue", "nengo_yana": "tab:red"}
            t = np.arange(results[list(results)[0]]["traj"].shape[0]) * 0.001
            for label, r in results.items():
                ntraj = normalize(r["traj"])
                axs[0].plot(t, ntraj, color=colors.get(label), alpha=0.7, lw=0.8)
                axs[1].plot(t, acquisition(case, ntraj), color=colors.get(label),
                            alpha=0.8, lw=1.2)
            axs[0].set_title(f"case '{case_name}': normalized solution SSP")
            axs[0].set_xlabel("time [s]")
            axs[0].set_ylabel(r"$\phi_t / \|\phi_t\|$")
            axs[1].set_title(f"case '{case_name}': acquisition along trajectory")
            axs[1].set_xlabel("time [s]")
            handles = [plt.Line2D([], [], color=colors[l], lw=2, label=l)
                       for l in results]
            fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
            fig.savefig(f"yana_toy_{case_name}.png", dpi=150)
            print(f"    saved yana_toy_{case_name}.png")


if __name__ == "__main__":
    main()
