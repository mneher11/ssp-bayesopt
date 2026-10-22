"""Resumable paper sweeps, with independent host workers sharing a YANA FPGA.

Example:
    python experiments/run_paper_experiments.py --workspace-dir . \
        --yana-host kria.local --workers 4 --shard-index 0 --num-shards 2

Use --dry-run to inspect coverage without creating artifacts or using hardware.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import subprocess
import sys
import threading

FUNCTIONS = ("himmelblau", "goldstein-price", "branin-hoo")
NEURONS = {25: tuple(range(5, 41, 5)), 55: tuple(range(2, 19, 2)), 121: tuple(range(1, 9))}
HEX_LAYOUTS = {25: (2, 2), 55: (3, 3), 121: (5, 4)}
RUNNER = Path(__file__).resolve().with_name("run_agent.py")
FIXED = dict(
    schema_version=1, agent_type="ssp-hex", num_samples=100,
    num_init_samples=10, nengo=True, backend="yana", beta_ucb=1.0,
    gamma=0.0, decay=False, len_scale=-1.0, sim_time=2.5, dt=0.001,
    tau=0.05, tau_ref=0.0, neuron_type="lif", decoder_method="direct-optim",
    num_restarts=1, rate=1.0, bias_free=False, power_mode="run_only",
    power_interval_ms=2.0, yana_readout_every=2500, fpga_clock_period_ns=5.0,
)


def sweep_configs(num_samples=100, clock_period_ns=5.0):
    """Union of the two plots, in a stable order used for shard assignment."""
    configs = []
    for function in FUNCTIONS:
        for dimension, neurons in NEURONS.items():
            for sparsity in ((0.0, 0.5, 0.9) if dimension == 55 else (0.0,)):
                for count in neurons:
                    assert count * dimension <= 1024
                    scales, rotates = HEX_LAYOUTS[dimension]
                    configs.append(dict(
                        FIXED, function_name=function, ssp_dim=dimension,
                        num_neurons=count, encoder_sparsity=sparsity,
                        n_scales=scales, n_rotates=rotates,
                        num_samples=num_samples, fpga_clock_period_ns=clock_period_ns,
                    ))
    return configs


def selected_configs(args):
    # Partition the FULL grid before filtering; subsets never reshuffle shards.
    return [c for index, c in enumerate(sweep_configs(args.num_samples, args.fpga_clock_period_ns))
            if index % args.num_shards == args.shard_index
            and (not args.functions or c['function_name'] in args.functions)
            and (not args.ssp_dims or c['ssp_dim'] in args.ssp_dims)
            and (not args.encoder_sparsities or c['encoder_sparsity'] in args.encoder_sparsities)
            and (not args.num_neurons or c['num_neurons'] in args.num_neurons)]


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def config_path(root, config):
    return root / config['function_name'] / (
        f"dim-{config['ssp_dim']}_sparsity-{config['encoder_sparsity']:g}"
        f"_neurons-{config['num_neurons']}"
    )


def read_json(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def exclusive_lock(path):
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another orchestrator owns {path}") from exc
        yield


def ensure_config(directory, config):
    path = directory / 'config.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError(f"Configuration mismatch at {path}; use a different workspace")
    else:
        atomic_json(path, config)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(directory, config, seed):
    """Check the actual trajectory and every paired power/timing capture."""
    import numpy as np

    trajectories = list((directory / 'trajectory').rglob('*.npz'))
    if len(trajectories) != 1:
        raise ValueError(f"Expected one trajectory in {directory}, got {len(trajectories)}")
    count = config['num_samples']
    expected = {k: config[k] for k in (
        'function_name', 'agent_type', 'ssp_dim', 'encoder_sparsity', 'num_neurons',
        'num_samples', 'num_init_samples', 'n_scales', 'n_rotates', 'nengo', 'backend',
        'beta_ucb', 'gamma', 'decay', 'len_scale', 'sim_time',
    )}
    expected.update(seed=seed, effective_ssp_dim=config['ssp_dim'])
    with np.load(trajectories[0], allow_pickle=False) as data:
        for key, value in expected.items():
            if data[key].item() != value:
                raise ValueError(f"Trajectory parameter mismatch: {key} in {directory}")
        size = count + config['num_init_samples']
        for name in ('vals', 'regret'):
            if data[name].shape != (size,) or not np.isfinite(data[name]).all():
                raise ValueError(f"Invalid {name} in {directory}")
        if data['sample_locs'].shape != (size, 2) or not np.isfinite(data['sample_locs']).all():
            raise ValueError(f"Invalid sample locations in {directory}")
        cycles = data['fpga_cycles'].copy()
        latencies = data['fpga_latencies_ns'].copy()
        if cycles.shape != (count,) or latencies.shape != (count,):
            raise ValueError(f"Incomplete FPGA timings in {directory}")
    measurements = sorted((directory / 'measurements').glob('sample-*.json'))
    if len(measurements) != count:
        raise ValueError(f"Expected {count} measurements in {directory}, got {len(measurements)}")
    for index, path in enumerate(measurements, 1):
        data = read_json(path)
        if (path.name != f'sample-{index:04d}.json' or data['seed'] != seed
                or data['optimization_sample'] != index
                or data['trajectory_index'] != config['num_init_samples'] + index - 1):
            raise ValueError(f"Measurement/trajectory association mismatch: {path}")
        if (data.get('error') or data['mode'] != config['power_mode'] or not data['samples']
                or data['power_interval_ms'] != config['power_interval_ms']):
            raise ValueError(f"Missing or failed run-only power capture: {path}")
        if any(row['phase'] != 'run' or not math.isfinite(row['power_uw'])
               or row['power_uw'] < 0 for row in data['samples']):
            raise ValueError(f"Invalid power samples: {path}")
        fpga = data['fpga']
        end = 0
        for window in fpga['windows']:
            if (window['start_timestep'] != end or window['end_timestep'] <= end
                    or not isinstance(window['cycles'], int) or window['cycles'] < 0):
                raise ValueError(f"Invalid execution window: {path}")
            end = window['end_timestep']
        total = sum(w['cycles'] for w in fpga['windows'])
        if (end != round(config['sim_time'] / config['dt']) or total <= 0
                or fpga['cycles'] != total or cycles[index - 1] != total
                or fpga['clock_period_ns'] != config['fpga_clock_period_ns']
                or fpga['latency_ns'] != total * config['fpga_clock_period_ns']
                or latencies[index - 1] != fpga['latency_ns']):
            raise ValueError(f"Invalid FPGA timing: {path}")
    return trajectories + measurements + [directory / 'trial.json']


def completed(directory, config):
    marker = directory / 'complete.json'
    if not marker.exists():
        return False
    metadata = read_json(directory / 'trial.json')
    saved = read_json(marker)
    if (metadata['config_id'] != config_id(config)
            or directory.name != f"trial-{metadata['trial_index']:04d}"
            or saved['config_id'] != config_id(config) or saved['seed'] != metadata['seed']):
        raise ValueError(f"Completed trial metadata mismatch: {directory}")
    files = validate_artifacts(directory, config, metadata['seed'])
    actual = {str(path.relative_to(directory)): checksum(path) for path in files}
    if saved['files'] != actual:
        raise ValueError(f"Completed trial checksum mismatch: {directory}")
    return True


def mark_complete(directory, config, seed):
    files = validate_artifacts(directory, config, seed)
    atomic_json(directory / 'complete.json', dict(
        config_id=config_id(config), seed=seed, completed_at=now(),
        files={str(p.relative_to(directory)): checksum(p) for p in files},
    ))


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_trial(directory, config, index, host):
    seed = secrets.randbelow(2**31 - 2) + 1
    if directory.exists():
        if (directory / 'trial.json').exists():
            metadata = read_json(directory / 'trial.json')
            if metadata['config_id'] != config_id(config) or metadata['trial_index'] != index:
                raise ValueError(f"Incomplete trial metadata mismatch: {directory}")
            seed = metadata['seed']  # Retry the original random trial, not a replacement draw.
        # A crash before metadata was committed cannot have launched the runner.
        archive = directory.with_name(directory.name + '-attempts')
        archive.mkdir(exist_ok=True)
        number = 1
        while (archive / f'attempt-{number:04d}').exists():
            number += 1
        directory.rename(archive / f'attempt-{number:04d}')
    directory.mkdir()
    atomic_json(directory / 'trial.json', dict(
        config_id=config_id(config), trial_index=index, seed=seed,
        yana_host=host, started_at=now(),
    ))
    return seed


def runner_command(config, directory, seed, host, scratch):
    command = [sys.executable, '-u', str(RUNNER)]
    for key, flag in (
        ('function_name', '--func'), ('agent_type', '--agent'), ('num_samples', '--num-samples'),
        ('ssp_dim', '--ssp-dim'), ('n_scales', '--n-scales'), ('n_rotates', '--n-rotates'),
        ('encoder_sparsity', '--encoder-sparsity'), ('num_neurons', '--num-neurons'),
        ('beta_ucb', '--beta-ucb'), ('gamma', '--gamma'), ('len_scale', '--len-scale'),
        ('yana_readout_every', '--yana-readout-every'), ('power_mode', '--power-mode'),
        ('power_interval_ms', '--power-interval-ms'), ('fpga_clock_period_ns', '--fpga-clock-period-ns'),
    ):
        command += [flag, str(config[key])]
    command += ['--num-trials', '1', '--seed', str(seed), '--nengo', '--backend', 'yana',
                '--yana-host', host, '--data-dir', str(directory / 'trajectory'),
                '--power-output', str(directory / 'measurements' / 'power'),
                '--measurements-dir', str(directory / 'measurements'),
                '--failure-output', str(directory / 'failure.json'),
                '--yana-deployment-dir', str(scratch / 'deployment')]
    return command


def execute_trial(command, directory, scratch, stop):
    temporary = scratch / 'tmp'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    environment = dict(os.environ, TMPDIR=str(temporary))
    with (directory / 'run.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment,
                                   cwd=RUNNER.parent.parent, start_new_session=True)
        try:
            while True:
                try:
                    return process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    if stop.is_set():
                        raise InterruptedError('Experiment stopped')
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def run_configuration(config, root, scratch, args, emit, stop, execute=execute_trial):
    directory = config_path(root, config)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / '.lock'):
        ensure_config(directory, config)
        blocked_path = directory / 'capacity-blocked.json'
        blocked = read_json(blocked_path) if blocked_path.exists() else None
        if blocked and blocked['config_id'] != config_id(config):
            raise ValueError(f"Capacity record mismatch at {blocked_path}")
        for index in range(1, args.num_trials + 1):
            if stop.is_set():
                return
            trial = directory / f'trial-{index:04d}'
            label = f"{directory.parent.name}/{directory.name} trial {index}"
            if completed(trial, config):
                emit('skipped', label)
                continue
            if blocked:
                emit('capacity-blocked', label)
                continue
            # Recover a crash after all outputs were saved but before completion.
            if (trial / 'trial.json').exists():
                metadata = read_json(trial / 'trial.json')
                if metadata['config_id'] != config_id(config) or metadata['trial_index'] != index:
                    raise ValueError(f"Trial metadata mismatch: {trial}")
                try:
                    mark_complete(trial, config, metadata['seed'])
                except (ValueError, KeyError, OSError, EOFError):
                    pass
                else:
                    emit('skipped', label)
                    continue
            seed = prepare_trial(trial, config, index, args.yana_host)
            command = runner_command(config, trial, seed, args.yana_host, scratch)
            atomic_json(trial / 'command.json', command)
            emit('started', label)
            status = execute(command, trial, scratch, stop)
            if status != 0:
                failure = read_json(trial / 'failure.json') if (trial / 'failure.json').exists() else {}
                if (status == 2 and failure.get('kind') == 'capacity'
                        and failure.get('exception_type') == 'yana.deploy.accelerator.HardwareCapacityError'
                        and failure.get('seed') == seed and failure.get('violations')):
                    # A random network failed. Fail fast without pretending that
                    # every other seed is proven infeasible or replacing this seed.
                    blocked = dict(config_id=config_id(config), observed_at=now(),
                                   failed_trial=index, seed=seed, failure=failure,
                                   policy='remaining trials not run after observed capacity failure')
                    atomic_json(blocked_path, blocked)
                    emit('capacity-blocked', label)
                    continue
                raise RuntimeError(f"Runner exited {status}; inspect {trial / 'run.log'}")
            try:
                mark_complete(trial, config, seed)
            except (ValueError, KeyError, OSError) as exc:
                atomic_json(trial / 'validation-error.json', dict(message=str(exc)))
                raise RuntimeError(f"Artifact validation failed for {trial}: {exc}") from exc
            emit('completed', label)


def run(args, execute=execute_trial):
    from tqdm import tqdm

    configs = selected_configs(args)
    if not configs:
        raise ValueError('No configurations selected')
    total = len(configs) * args.num_trials
    if args.dry_run:
        print(json.dumps(dict(configurations=len(configs), trials=total,
                              shard_index=args.shard_index, num_shards=args.num_shards), indent=2))
        for config in configs:
            print(config_path(Path('<artifacts>/paper'), config))
        return
    root = args.workspace_dir.resolve() / 'artifacts' / 'paper'
    work = root / '.work' / f'shard-{args.shard_index}'
    work.mkdir(parents=True, exist_ok=True)
    pending, events, stop = queue.Queue(), queue.Queue(), threading.Event()
    for config in configs:
        pending.put(config)

    def worker(index):
        scratch = work / f'worker-{index}'
        scratch.mkdir(exist_ok=True)
        while not stop.is_set():
            try:
                config = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_configuration(config, root, scratch, args,
                                  lambda kind, label: events.put((kind, label)), stop, execute)
            except Exception as exc:
                stop.set()
                events.put(('error', str(exc)))
                return

    counts = dict(completed=0, skipped=0, **{'capacity-blocked': 0})
    with exclusive_lock(work / 'orchestrator.lock'):
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(worker, index) for index in range(args.workers)]
        error = None
        try:
            with tqdm(total=total, unit='trial') as progress:
                while progress.n < total:
                    try:
                        kind, label = events.get(timeout=0.2)
                    except queue.Empty:
                        if all(f.done() for f in futures):
                            for future in futures:
                                future.result()
                            raise RuntimeError('Workers stopped before all trials were accounted for')
                        continue
                    progress.set_description(label)
                    if kind == 'error':
                        error = label
                        break
                    if kind in counts:
                        counts[kind] += 1
                        progress.update(1)
                        progress.set_postfix(counts)
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
        if error:
            raise RuntimeError(error)
    print(json.dumps(counts, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-dir', type=Path, default=Path.cwd())
    parser.add_argument('--yana-host', required=True)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1,
                        help='concurrent host workers sharing this FPGA (default: 1)')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--functions', nargs='+', choices=FUNCTIONS)
    parser.add_argument('--ssp-dims', nargs='+', type=int, choices=tuple(NEURONS))
    parser.add_argument('--encoder-sparsities', nargs='+', type=float, choices=(0.0, 0.5, 0.9))
    parser.add_argument('--num-neurons', nargs='+', type=int)
    parser.add_argument('--fpga-clock-period-ns', type=float, default=5.0,
                        help='actual deployed FPGA clock period (default: 5 ns / 200 MHz)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.num_trials, args.num_samples, args.workers, args.num_shards) < 1:
        parser.error('trial, sample, worker and shard counts must be positive')
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('--shard-index must be in [0, --num-shards)')
    if not math.isfinite(args.fpga_clock_period_ns) or args.fpga_clock_period_ns <= 0:
        parser.error('--fpga-clock-period-ns must be finite and positive')
    return args


if __name__ == '__main__':
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print('Interrupted; incomplete attempts will be preserved on resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'Experiment stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
