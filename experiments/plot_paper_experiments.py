"""Read-only snapshots of the running paper sweep's energy/regret trade-offs.

Only trials with a valid, checksummed complete.json are used. No locks, recovery,
FPGA connections, imports of the runner, or writes to input trees are performed.

Example (one shared tree or two locally accessible shard trees):
    python experiments/plot_paper_experiments.py \
        --artifact-dirs artifacts/paper /mnt/other-host/artifacts/paper \
        --output-dir /tmp/ssp-paper-plots
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

FUNCTIONS = ('himmelblau', 'goldstein-price', 'branin-hoo')
TITLES = ('Himmelblau', 'Goldstein–Price', 'Branin–Hoo')
DIMENSIONS = (25, 55, 121)
OUTPUTS = ('plot1-dimensions.png', 'plot2-sparsity.png', 'summary.csv')
DEFAULT_HOST_POWER_OFFSETS_MW = {'kria.local': 2360.0, 'ess-kria-03': 2800.0}


@dataclass
class TrialMetrics:
    config: dict
    config_id: str
    trial_index: int
    seed: int
    mean_energy_j: float
    final_average_regret: float
    signature: tuple
    path: Path


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def inside(path, parent):
    return path == parent or parent in path.parents


def verified_bytes(directory, relative, manifest):
    """Read once, then verify those exact bytes against the completion marker."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        raise ValueError(f'Unsafe manifest path: {relative}')
    path = (directory / relative_path).resolve()
    if not inside(path, directory.resolve()):
        raise ValueError(f'Manifest path escapes trial directory: {relative}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[relative]:
        raise ValueError(f'Checksum mismatch: {relative}')
    return data


def read_trial(marker, host_power_offsets_mw=None):
    directory = marker.parent
    config = json.loads((directory.parent / 'config.json').read_bytes())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported artifact schema')
    saved = json.loads(marker.read_bytes())
    identity = digest_config(config)
    if saved['config_id'] != identity:
        raise ValueError('Configuration fingerprint mismatch')
    manifest = saved['files']
    if not isinstance(manifest, dict) or any(not isinstance(p, str) or not Path(p).parts for p in manifest):
        raise ValueError('Invalid completion manifest')
    metadata = json.loads(verified_bytes(directory, 'trial.json', manifest))
    if (metadata['config_id'] != identity or metadata['seed'] != saved['seed']
            or directory.name != f"trial-{metadata['trial_index']:04d}"):
        raise ValueError('Trial identity mismatch')
    count = config['num_samples']
    initial = config['num_init_samples']
    if not isinstance(count, int) or count < 1 or not isinstance(initial, int) or initial < 0:
        raise ValueError('Invalid sample counts')
    if (config['backend'] != 'yana' or config['power_mode'] != 'run_only'
            or config['function_name'] not in FUNCTIONS or config['ssp_dim'] not in DIMENSIONS
            or config['encoder_sparsity'] not in (0.0, 0.5, 0.9)
            or (config['encoder_sparsity'] != 0 and config['ssp_dim'] != 55)):
        raise ValueError('Not a supported run-only YANA paper trial')
    trajectories = [p for p in manifest if Path(p).parts[0] == 'trajectory' and p.endswith('.npz')]
    measurements = [f'measurements/sample-{i:04d}.json' for i in range(1, count + 1)]
    if len(trajectories) != 1 or set(manifest) != {'trial.json', *trajectories, *measurements}:
        raise ValueError('Completion manifest does not contain the expected trial files')
    with np.load(BytesIO(verified_bytes(directory, trajectories[0], manifest)), allow_pickle=False) as data:
        for key in ('function_name', 'agent_type', 'ssp_dim', 'encoder_sparsity', 'num_neurons',
                    'num_samples', 'num_init_samples', 'n_scales', 'n_rotates', 'backend',
                    'nengo', 'beta_ucb', 'gamma', 'decay', 'len_scale', 'sim_time'):
            if data[key].item() != config[key]:
                raise ValueError(f'Trajectory parameter mismatch: {key}')
        if data['seed'].item() != saved['seed'] or data['effective_ssp_dim'].item() != config['ssp_dim']:
            raise ValueError('Trajectory seed/dimension mismatch')
        regrets = np.asarray(data['regret'], dtype=float)
        if regrets.shape != (count + initial,) or not np.isfinite(regrets).all():
            raise ValueError('Incomplete or invalid regret trajectory')
        cycles = data['fpga_cycles'].copy()
        latencies = data['fpga_latencies_ns'].copy()
        if cycles.shape != (count,) or latencies.shape != (count,):
            raise ValueError('Incomplete trajectory FPGA timings')
        final_regret = float(np.mean(regrets))
        if not math.isfinite(final_regret):
            raise ValueError('Nonfinite final average regret')
    if host_power_offsets_mw is None:
        host_power_offsets_mw = DEFAULT_HOST_POWER_OFFSETS_MW
    host = metadata.get('yana_host')
    # Unknown hosts retain the raw measurement; callers can explicitly supply
    # an offset (including zero) for them via --host-power-offset-mw.
    offset_uw = host_power_offsets_mw.get(host, 0.0) * 1000.0
    energies = []
    for index, relative in enumerate(measurements, 1):
        data = json.loads(verified_bytes(directory, relative, manifest))
        if (data['seed'] != saved['seed'] or data['optimization_sample'] != index
                or data['trajectory_index'] != initial + index - 1):
            raise ValueError(f'Wrong sample/trajectory association: {relative}')
        if (data.get('error') or data['mode'] != 'run_only' or not data['samples']
                or data['power_interval_ms'] != config['power_interval_ms']):
            raise ValueError(f'Invalid run-only power capture: {relative}')
        powers = [row['power_uw'] for row in data['samples']]
        if any(row['phase'] != 'run' for row in data['samples']) or any(
                not math.isfinite(power) or power < 0 for power in powers):
            raise ValueError(f'Invalid run power readings: {relative}')
        timing = data['fpga']
        clock = config['fpga_clock_period_ns']
        if not math.isfinite(clock) or clock <= 0:
            raise ValueError('Invalid FPGA clock period')
        end = 0
        total_cycles = 0
        for window in timing['windows']:
            if (window['start_timestep'] != end or window['end_timestep'] <= end
                    or not isinstance(window['cycles'], int) or window['cycles'] < 0):
                raise ValueError(f'Invalid FPGA execution window: {relative}')
            end = window['end_timestep']
            total_cycles += window['cycles']
        latency_ns = total_cycles * clock
        if (total_cycles <= 0 or end != round(config['sim_time'] / config['dt'])
                or timing['cycles'] != total_cycles or timing['clock_period_ns'] != clock
                or timing['latency_ns'] != latency_ns or cycles[index - 1] != total_cycles
                or latencies[index - 1] != latency_ns):
            raise ValueError(f'FPGA timing mismatch: {relative}')
        # Average within each sample first, not across all sensor readings of
        # the trial: samples with more readings must not receive extra weight.
        # Measurements are in µW; host offsets are specified in mW. Clamp
        # below-zero residual readings to zero before converting power to energy.
        corrected_power_uw = np.maximum(np.asarray(powers, dtype=float) - offset_uw, 0.0)
        energies.append(float(np.mean(corrected_power_uw)) * 1e-6 * latency_ns * 1e-9)
    energy = float(np.mean(energies))
    if not math.isfinite(energy) or energy < 0:
        raise ValueError('Nonfinite inference energy')
    return TrialMetrics(config, identity, metadata['trial_index'], saved['seed'],
                        energy, final_regret, tuple(sorted(manifest.items())), directory)


def load_trials(roots, warn=None, host_power_offsets_mw=None):
    """Snapshot marker paths once; new/in-progress trials wait for the next run."""
    if warn is None:
        warn = lambda message: print(message, file=sys.stderr)
    found = []
    for root in roots:
        found.extend((root, marker) for marker in sorted(root.glob('*/*/trial-*/complete.json'))
                     if re.fullmatch(r'trial-\d+', marker.parent.name))
    unique = {}
    skipped = 0
    for root, marker in found:
        try:
            if not inside(marker.resolve(), root.resolve()):
                raise ValueError('Completion marker escapes input tree')
            trial = read_trial(marker, host_power_offsets_mw)
        except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
            warn(f'[skip] {marker.parent}: {exc}')
            skipped += 1
            continue
        key = trial.config_id, trial.trial_index
        previous = unique.get(key)
        if previous is not None:
            if previous.seed != trial.seed or previous.signature != trial.signature:
                raise ValueError(f'Conflicting copies of the same trial: {previous.path} and {trial.path}')
            continue
        unique[key] = trial
    print(f'Found {len(found)} completion markers; loaded {len(unique)} unique completed trials; '
          f'skipped {skipped} invalid/unavailable trials.')
    return list(unique.values())


def aggregate(trials):
    """One point per configuration, with equal weight for each completed trial."""
    groups = {}
    fixed_reference = None
    point_ids = {}
    variable_keys = {'function_name', 'ssp_dim', 'encoder_sparsity', 'num_neurons', 'n_scales', 'n_rotates'}
    for trial in trials:
        config = trial.config
        fixed = {k: v for k, v in config.items() if k not in variable_keys}
        if fixed_reference is None:
            fixed_reference = fixed
        elif fixed != fixed_reference:
            raise ValueError('Input trials have different fixed settings (e.g. budgets, agent, or clock). '
                             'Plot these experiment versions separately.')
        point_key = tuple(config[k] for k in ('function_name', 'ssp_dim', 'encoder_sparsity', 'num_neurons'))
        if point_key in point_ids and point_ids[point_key] != trial.config_id:
            raise ValueError(f'Conflicting parameter sets for plot point {point_key}')
        point_ids[point_key] = trial.config_id
        groups.setdefault(trial.config_id, []).append(trial)
    rows = []
    for identity, group in groups.items():
        config = group[0].config
        row = {k: config[k] for k in ('function_name', 'ssp_dim', 'encoder_sparsity', 'num_neurons')}
        row.update(config_id=identity, completed_trials=len(group),
                   mean_inference_energy_j=float(np.mean([t.mean_energy_j for t in group])),
                   final_cumulative_average_regret=float(np.mean([t.final_average_regret for t in group])))
        rows.append(row)
    return sorted(rows, key=lambda r: (FUNCTIONS.index(r['function_name']), r['ssp_dim'],
                                       r['encoder_sparsity'], r['num_neurons']))


def output_directory(path, roots):
    destination = path.resolve()
    for root in roots:
        if inside(destination, root.resolve()):
            raise ValueError(f'Output directory must be outside every input tree: {destination}')
        for name in OUTPUTS:
            if inside((destination / name).resolve(), root.resolve()):
                raise ValueError(f'Output file points into an input tree: {destination / name}')
    return destination


def atomic_output(path, writer):
    """Replace outputs atomically; never follow a pre-existing output hardlink."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix='.plot-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def draw_plot(rows, kind, path, annotate=True):
    import matplotlib.pyplot as plt

    if kind == 'dimensions':
        series = [(dimension, f'{dimension} dimensions', color, marker)
                  for dimension, color, marker in zip(DIMENSIONS, ('#0072B2', '#D55E00', '#009E73'), ('o', 's', '^'))]
        title = 'Plot 1: dense encoders, varying SSP dimension'
    else:
        series = [(sparsity, f'{sparsity:.0%} sparsity', color, marker)
                  for sparsity, color, marker in ((0.5, '#CC79A7', 'o'), (0.9, '#D55E00', 's'))]
        title = 'Plot 2: 55 dimensions, varying encoder sparsity'
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), squeeze=False)
    try:
        for ax, function, label in zip(axes[0], FUNCTIONS, TITLES):
            plotted = False
            for value, legend, color, marker in series:
                points = [r for r in rows if r['function_name'] == function and (
                    (kind == 'dimensions' and r['ssp_dim'] == value and r['encoder_sparsity'] == 0)
                    or (kind == 'sparsity' and r['ssp_dim'] == 55 and r['encoder_sparsity'] == value))]
                points.sort(key=lambda row: row['num_neurons'])
                if not points:
                    continue
                x = [r['mean_inference_energy_j'] * 1e3 for r in points]
                y = [r['final_cumulative_average_regret'] for r in points]
                ax.plot(x, y, color=color, marker=marker, linewidth=1.2, markersize=6, label=legend)
                if annotate:
                    for index, (px, py, point) in enumerate(zip(x, y, points)):
                        ax.annotate(f"{point['num_neurons']} ({point['completed_trials']})", (px, py),
                                    xytext=(5, 7 if index % 2 == 0 else -13),
                                    textcoords='offset points', color=color, fontsize=7)
                plotted = True
            ax.set_title(label)
            ax.set_xlabel('Mean PL inference energy [mJ] (host PS offset subtracted)')
            ax.set_ylabel('Final cumulative average regret')
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 4), useOffset=False)
            ax.grid(alpha=0.25)
            ax.margins(x=0.2, y=0.2)
            if plotted:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No completed trials yet', transform=ax.transAxes, ha='center', va='center')
        fig.suptitle(title + ' — completed trials only')
        note = 'Lines follow increasing neurons/dimension; lower-left is better.'
        if annotate:
            note = 'Point labels: neurons/dimension (completed trials). ' + note
        fig.text(0.5, 0.025, note, ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.065, 1, 0.94))
        atomic_output(path, lambda stream: fig.savefig(stream, format='png', dpi=200))
    finally:
        plt.close(fig)


def save_plots(rows, destination, annotate=True):
    destination.mkdir(parents=True, exist_ok=True)
    # Keep Matplotlib's font/config cache outside ALL running artifact trees,
    # even if an inherited MPLCONFIGDIR points into one of those trees.
    previous = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='.matplotlib-', dir=destination) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            with matplotlib.rc_context({'text.usetex': False}):
                draw_plot(rows, 'dimensions', destination / OUTPUTS[0], annotate)
                draw_plot(rows, 'sparsity', destination / OUTPUTS[1], annotate)
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous
    # CSV is an audit-friendly copy of exactly the points plotted (energy in J).
    columns = ['function_name', 'ssp_dim', 'encoder_sparsity', 'num_neurons', 'completed_trials',
               'mean_inference_energy_j', 'final_cumulative_average_regret', 'config_id']
    def write_csv(stream):
        import io
        wrapper = io.TextIOWrapper(stream, encoding='utf-8', newline='', write_through=True)
        try:
            writer = csv.DictWriter(wrapper, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            wrapper.flush()
        finally:
            wrapper.detach()
    atomic_output(destination / OUTPUTS[2], write_csv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dirs', type=Path, nargs='+',
                        default=[Path(__file__).resolve().parents[1] / 'artifacts' / 'paper'],
                        help='one shared artifact tree or multiple locally accessible shard trees (read only)')
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/ssp-paper-plots'))
    parser.add_argument('--no-annotations', action='store_true', help='omit neuron/trial-count labels')
    parser.add_argument('--host-power-offset-mw', action='append', default=[], metavar='HOST=MW',
                        help='override/add a static host power offset in mW; repeatable '
                             '(defaults: kria.local=2360, ess-kria-03=2800; use HOST=0 to disable)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    roots = list(dict.fromkeys(path.resolve() for path in args.artifact_dirs))
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'Artifact directory does not exist: {root}')
    destination = output_directory(args.output_dir, roots)
    offsets = dict(DEFAULT_HOST_POWER_OFFSETS_MW)
    for value in args.host_power_offset_mw:
        if '=' not in value:
            raise ValueError(f'Invalid --host-power-offset-mw {value!r}; expected HOST=MW')
        host, raw_offset = value.split('=', 1)
        try:
            offset = float(raw_offset)
        except ValueError as exc:
            raise ValueError(f'Invalid power offset for {host!r}: {raw_offset!r}') from exc
        if not host or not math.isfinite(offset) or offset < 0:
            raise ValueError(f'Invalid power offset {value!r}; host and nonnegative finite MW required')
        offsets[host] = offset
    trials = load_trials(roots, host_power_offsets_mw=offsets)
    rows = aggregate(trials)
    save_plots(rows, destination, annotate=not args.no_annotations)
    print(f'Plotted {len(rows)} configurations from {len(trials)} completed trials.')
    for name in OUTPUTS:
        print(destination / name)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f'Plotting stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
