"""Paper coverage, artifact validation and resume without requiring an FPGA."""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_paper_experiments as paper


@pytest.fixture
def config():
    return paper.sweep_configs(num_samples=2)[0]


def options(tmp_path, **kwargs):
    args = paper.parse_args(['--workspace-dir', str(tmp_path), '--yana-host', 'fake'])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def fake_artifacts(directory, config, seed):
    (directory / 'trajectory').mkdir(exist_ok=True)
    (directory / 'measurements').mkdir(exist_ok=True)
    size = config['num_samples'] + config['num_init_samples']
    cycles = np.arange(1, config['num_samples'] + 1) * 100
    np.savez(directory / 'trajectory' / 'result.npz', **config, seed=seed,
             effective_ssp_dim=config['ssp_dim'], vals=np.zeros(size), regret=np.ones(size),
             sample_locs=np.zeros((size, 2)), fpga_cycles=cycles,
             fpga_latencies_ns=cycles * config['fpga_clock_period_ns'])
    for index, count in enumerate(cycles, 1):
        paper.atomic_json(directory / 'measurements' / f'sample-{index:04d}.json', dict(
            seed=seed, optimization_sample=index,
            trajectory_index=config['num_init_samples'] + index - 1,
            mode='run_only', power_interval_ms=2.0, error=None,
            samples=[dict(timestamp_ns=100, power_uw=123456, phase='run')],
            markers=[dict(timestamp_ns=90, phase='run', event='begin'),
                     dict(timestamp_ns=110, phase='run', event='end')],
            fpga=dict(windows=[dict(start_timestep=0, end_timestep=2500, cycles=int(count))],
                      cycles=int(count), clock_period_ns=config['fpga_clock_period_ns'],
                      latency_ns=int(count) * config['fpga_clock_period_ns']),
        ))


def execute_success(config, calls):
    def execute(command, directory, scratch, stop):
        seed = int(command[command.index('--seed') + 1])
        calls.append((directory, seed, scratch))
        fake_artifacts(directory, config, seed)
        return 0
    return execute


def run_config(config, tmp_path, args, execute):
    events = []
    paper.run_configuration(config, tmp_path, tmp_path / 'scratch', args,
                            lambda *event: events.append(event), threading.Event(), execute)
    return events


def test_exact_union_and_real_dimensions():
    grid = paper.sweep_configs()
    assert len(grid) == 129
    assert len(grid) * 5 == 645
    assert len({paper.config_id(c) for c in grid}) == len(grid)
    for function in paper.FUNCTIONS:
        subset = [c for c in grid if c['function_name'] == function]
        assert len(subset) == 43
        assert sum(c['encoder_sparsity'] == 0 for c in subset) == 25
        assert sum(c['encoder_sparsity'] > 0 for c in subset) == 18
    for c in grid:
        assert c['ssp_dim'] * c['num_neurons'] <= 1024
        assert c['ssp_dim'] == 6 * c['n_scales'] * c['n_rotates'] + 1
        assert c['encoder_sparsity'] == 0 or c['ssp_dim'] == 55


def test_shards_are_disjoint_complete_and_independent_of_trial_count(tmp_path):
    args = options(tmp_path, num_shards=2)
    first = {paper.config_id(c) for c in paper.selected_configs(args)}
    args.shard_index = 1
    second = {paper.config_id(c) for c in paper.selected_configs(args)}
    assert not first & second
    assert first | second == {paper.config_id(c) for c in paper.sweep_configs()}
    args.num_trials = 10
    assert second == {paper.config_id(c) for c in paper.selected_configs(args)}
    args.ssp_dims = [55]
    assert {paper.config_id(c) for c in paper.selected_configs(args)} <= second


def test_dry_run_creates_no_artifacts(tmp_path, capsys):
    paper.run(options(tmp_path, dry_run=True))
    assert '"trials": 645' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_resume_and_increasing_trial_count_keep_all_completed_bytes(tmp_path, config, monkeypatch):
    random_draws = iter([17, 42, 123])
    monkeypatch.setattr(paper.secrets, 'randbelow', lambda _: next(random_draws))
    calls = []
    args = options(tmp_path, num_trials=2)
    execute = execute_success(config, calls)
    events = run_config(config, tmp_path, args, execute)
    assert [e[0] for e in events] == ['started', 'completed'] * 2
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert [e[0] for e in run_config(config, tmp_path, args, execute)] == ['skipped'] * 2
    assert len(calls) == 2
    args.num_trials = 3
    run_config(config, tmp_path, args, execute)
    assert len(calls) == 3
    assert all(p.read_bytes() == data for p, data in snapshots.items())
    assert [seed for _, seed, _ in calls] == [18, 43, 124]


def test_retry_preserves_incomplete_attempt_and_reuses_random_seed(tmp_path, config):
    args = options(tmp_path, num_trials=1)
    directory = paper.config_path(tmp_path, config)
    directory.mkdir(parents=True)
    paper.ensure_config(directory, config)
    trial = directory / 'trial-0001'
    seed = paper.prepare_trial(trial, config, 1, 'old-host')
    (trial / 'partial.txt').write_text('keep me')
    calls = []
    run_config(config, tmp_path, args, execute_success(config, calls))
    assert calls[0][1] == seed
    archive = directory / 'trial-0001-attempts' / 'attempt-0001'
    assert (archive / 'partial.txt').read_text() == 'keep me'
    assert paper.read_json(archive / 'trial.json')['yana_host'] == 'old-host'
    assert paper.completed(trial, config)


def test_crash_before_metadata_is_archived_and_resumable(tmp_path, config):
    args = options(tmp_path, num_trials=1)
    trial = paper.config_path(tmp_path, config) / 'trial-0001'
    trial.mkdir(parents=True)
    (trial / 'trial.json.tmp').write_text('interrupted write')
    calls = []
    run_config(config, tmp_path, args, execute_success(config, calls))
    archive = trial.with_name('trial-0001-attempts') / 'attempt-0001'
    assert (archive / 'trial.json.tmp').read_text() == 'interrupted write'
    assert paper.completed(trial, config)


def test_completed_trial_cannot_be_relabelled_as_another_repetition(tmp_path, config):
    args = options(tmp_path, num_trials=1)
    calls = []
    run_config(config, tmp_path, args, execute_success(config, calls))
    trial = calls[0][0]
    moved = trial.with_name('trial-0002')
    trial.rename(moved)
    with pytest.raises(ValueError, match='metadata mismatch'):
        paper.completed(moved, config)


def test_recovers_outputs_saved_before_completion_marker(tmp_path, config):
    args = options(tmp_path, num_trials=1)
    parent = paper.config_path(tmp_path, config)
    parent.mkdir(parents=True)
    trial = parent / 'trial-0001'
    seed = paper.prepare_trial(trial, config, 1, 'fake')
    fake_artifacts(trial, config, seed)
    calls = []
    events = run_config(config, tmp_path, args, execute_success(config, calls))
    assert calls == []
    assert events[0][0] == 'skipped'
    assert paper.completed(trial, config)


def test_configuration_mismatch_is_never_overwritten(tmp_path, config):
    calls = []
    args = options(tmp_path, num_trials=1)
    run_config(config, tmp_path, args, execute_success(config, calls))
    changed = dict(config, num_samples=3)
    with pytest.raises(ValueError, match='Configuration mismatch'):
        run_config(changed, tmp_path, args, execute_success(changed, calls))
    assert len(calls) == 1


@pytest.mark.parametrize('damage', ['power', 'timing', 'association', 'trajectory', 'checksum'])
def test_completed_artifacts_are_validated_without_overwrite(tmp_path, config, damage):
    args = options(tmp_path, num_trials=1)
    calls = []
    run_config(config, tmp_path, args, execute_success(config, calls))
    trial = calls[0][0]
    measurement = trial / 'measurements' / 'sample-0001.json'
    data = paper.read_json(measurement)
    if damage == 'power':
        data['samples'] = []
    elif damage == 'timing':
        data['fpga']['cycles'] += 1
    elif damage == 'association':
        data['trajectory_index'] += 1
    elif damage == 'trajectory':
        (trial / 'trajectory' / 'result.npz').unlink()
    elif damage == 'checksum':
        data['samples'][0]['power_uw'] += 1
    paper.atomic_json(measurement, data)
    with pytest.raises(ValueError):
        run_config(config, tmp_path, args, execute_success(config, calls))
    assert len(calls) == 1
    assert (trial / 'complete.json').exists()


def test_capacity_is_terminal_and_not_confused_with_other_errors(tmp_path, config):
    args = options(tmp_path, num_trials=3)
    calls = []

    def capacity(command, trial, scratch, stop):
        calls.append(trial)
        paper.atomic_json(trial / 'failure.json', dict(
            kind='capacity', exception_type='yana.deploy.accelerator.HardwareCapacityError',
            seed=paper.read_json(trial / 'trial.json')['seed'], violations=['synapse RAM full'],
        ))
        return 2

    events = run_config(config, tmp_path, args, capacity)
    assert [e[0] for e in events] == ['started'] + ['capacity-blocked'] * 3
    assert len(calls) == 1
    assert [e[0] for e in run_config(config, tmp_path, args, capacity)] == ['capacity-blocked'] * 3
    assert len(calls) == 1
    assert not (calls[0] / 'complete.json').exists()
    assert 'not run' in paper.read_json(calls[0].parent / 'capacity-blocked.json')['policy']


def test_unexpected_error_stops_without_capacity_exclusion(tmp_path, config):
    args = options(tmp_path, num_trials=2)

    def fail(command, trial, scratch, stop):
        paper.atomic_json(trial / 'failure.json', dict(kind='execution', message='connection lost'))
        return 1

    with pytest.raises(RuntimeError, match='Runner exited 1'):
        run_config(config, tmp_path, args, fail)
    assert not (paper.config_path(tmp_path, config) / 'capacity-blocked.json').exists()


def test_exit_zero_without_valid_outputs_is_not_a_completed_trial(tmp_path, config):
    args = options(tmp_path, num_trials=1)
    with pytest.raises(RuntimeError, match='Artifact validation failed'):
        run_config(config, tmp_path, args, lambda *a: 0)
    trial = paper.config_path(tmp_path, config) / 'trial-0001'
    assert (trial / 'validation-error.json').exists()
    assert not (trial / 'complete.json').exists()


def test_command_has_explicit_parameters_and_trial_isolation(tmp_path, config):
    command = paper.runner_command(config, tmp_path / 'trial', 123, 'board', tmp_path / 'worker')
    for flag, expected in [('--num-trials', '1'), ('--seed', '123'), ('--encoder-sparsity', '0.0'),
                           ('--power-mode', 'run_only'), ('--power-interval-ms', '2.0'),
                           ('--n-scales', '2'), ('--n-rotates', '2')]:
        assert command[command.index(flag) + 1] == expected
    assert str(tmp_path / 'worker' / 'deployment') in command
    assert str(tmp_path / 'trial' / 'measurements') in command


def test_concurrent_workers_get_distinct_reusable_scratch(tmp_path):
    args = options(tmp_path, num_trials=2, num_samples=2, workers=2,
                   functions=['himmelblau'], ssp_dims=[25], num_neurons=[5, 10])
    seen = []
    barrier = threading.Barrier(2)
    lock = threading.Lock()

    def execute(command, trial, scratch, stop):
        config = paper.read_json(trial.parent / 'config.json')
        seed = paper.read_json(trial / 'trial.json')['seed']
        with lock:
            seen.append((scratch, config['num_neurons']))
        barrier.wait(timeout=5)
        fake_artifacts(trial, config, seed)
        return 0

    paper.run(args, execute)
    assert len(seen) == 4
    assert len({scratch for scratch, _ in seen}) == 2
    for neurons in (5, 10):
        assert len({scratch for scratch, n in seen if n == neurons}) == 1


def test_real_subprocess_uses_and_clears_worker_tmp(tmp_path):
    trial, scratch = tmp_path / 'trial', tmp_path / 'scratch'
    trial.mkdir()
    scratch.mkdir()
    command = [sys.executable, '-c',
               'import os, pathlib; p=pathlib.Path(os.environ["TMPDIR"]); '
               'assert not (p / "old").exists(); (p / "old").write_text("x"); print(p)']
    for _ in range(2):
        assert paper.execute_trial(command, trial, scratch, threading.Event()) == 0
    assert (trial / 'run.log').read_text().strip() == str(scratch / 'tmp')


def test_stop_terminates_running_subprocess(tmp_path):
    trial, scratch = tmp_path / 'trial', tmp_path / 'scratch'
    trial.mkdir()
    scratch.mkdir()
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        paper.execute_trial([sys.executable, '-c', 'import time; time.sleep(60)'],
                            trial, scratch, stop)


def test_private_trial_cache_bypasses_corrupt_shared_nengo_entry(tmp_path, monkeypatch):
    import nengo
    from nengo.exceptions import CacheIOError
    from experiments import run_agent as runner

    shared = tmp_path / 'shared-cache'
    settings = nengo.rc['decoder_cache']
    for key, value in dict(enabled='True', readonly='False', path=str(shared)).items():
        monkeypatch.setitem(settings, key, value)
    with nengo.Network(seed=123) as network:
        stim = nengo.Node([0.5])
        neurons = nengo.Ensemble(20, 1)
        nengo.Connection(stim, neurons)
        probe = nengo.Probe(neurons, synapse=0.01)
    with nengo.Simulator(network, progress_bar=False) as sim:
        sim.run_steps(20)
        expected = sim.data[probe].copy()
    entries = list(shared.rglob('*.nco'))
    assert entries, 'regression must exercise an actual disk cache hit'
    for path in entries:
        with path.open('r+b') as stream:
            stream.write(b'BAD')  # Preserve file length, but corrupt the NCO magic header.
    with pytest.raises(CacheIOError, match='Not a Nengo cache object file'):
        nengo.Simulator(network, progress_bar=False)
    corrupted = {p: p.read_bytes() for p in entries}
    with runner.isolated_decoder_cache() as private:
        assert private != shared
        with nengo.Simulator(network, progress_bar=False) as sim:
            sim.run_steps(20)
            np.testing.assert_allclose(sim.data[probe], expected)
        assert list(private.rglob('*.nco'))  # Within-trial reuse is still enabled.
    assert not private.exists()
    assert settings['path'] == str(shared)
    assert all(path.read_bytes() == content for path, content in corrupted.items())


def test_private_trial_cache_cleans_up_and_restores_settings_after_failure(tmp_path, monkeypatch):
    from experiments import run_agent as runner

    settings = runner.nengo.rc['decoder_cache']
    shared = tmp_path / 'shared-cache'
    shared.mkdir()
    sentinel = shared / 'do-not-delete'
    sentinel.write_text('existing shared cache')
    for key, value in dict(enabled='True', readonly='True', path=str(shared)).items():
        monkeypatch.setitem(settings, key, value)
    with pytest.raises(RuntimeError, match='trial failed'):
        with runner.isolated_decoder_cache() as private:
            assert settings['path'] == str(private)
            assert not settings.getboolean('readonly')
            (private / 'partial-cache').write_text('incomplete')
            raise RuntimeError('trial failed')
    assert not private.exists()
    assert settings['path'] == str(shared)
    assert settings.getboolean('readonly')
    assert sentinel.read_text() == 'existing shared cache'


def test_private_trial_cache_respects_explicitly_disabled_cache(monkeypatch):
    from experiments import run_agent as runner

    settings = runner.nengo.rc['decoder_cache']
    monkeypatch.setitem(settings, 'enabled', 'False')
    previous = dict(settings)
    with runner.isolated_decoder_cache() as private:
        assert private is None
        assert dict(settings) == previous
    assert dict(settings) == previous


def test_runner_uses_fresh_worker_local_decoder_cache_for_each_yana_trial(tmp_path, monkeypatch):
    from experiments import run_agent as runner
    from nengo.cache import get_default_decoder_cache

    worker_tmp = tmp_path / 'worker-tmp'
    worker_tmp.mkdir()
    monkeypatch.setattr(runner.tempfile, 'tempdir', str(worker_tmp))
    settings = runner.nengo.rc['decoder_cache']
    for key, value in dict(enabled='True', readonly='True', path=str(tmp_path / 'shared')).items():
        monkeypatch.setitem(settings, key, value)
    paths = []

    def trial(self, **kwargs):
        cache = get_default_decoder_cache()
        path = cache.cache_dir
        assert path.parent == worker_tmp
        assert not (path / 'previous-trial').exists()
        (path / 'previous-trial').write_text('this trial only')
        assert not cache.readonly
        paths.append(path)
        return {}

    monkeypatch.setattr(runner.SamplingTrial, 'run', trial)
    runner.main(['--nengo', '--backend', 'yana', '--num-trials', '2', '--seed', '123',
                 '--data-dir', str(tmp_path / 'data')])
    assert len(set(paths)) == 2
    assert all(not path.exists() for path in paths)
    assert settings['path'] == str(tmp_path / 'shared')
    assert settings.getboolean('readonly')
    assert not (tmp_path / 'shared').exists()


def test_runner_forwards_sparsity_and_network_seed(monkeypatch):
    from experiments import run_agent as runner

    captured = {}

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def maximize(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop after forwarding')

    monkeypatch.setattr(runner.ssp_bayes_opt, 'NengoBayesianOptimization', FakeOptimizer)
    params = runner.SamplingTrial()._create_parameters(nengo=True, seed=123, encoder_sparsity=0.9)
    with pytest.raises(RuntimeError, match='stop after forwarding'):
        runner.SamplingTrial().evaluate(params)
    assert captured['encoder_sparsity'] == 0.9
    assert captured['random_state'] == 123
    assert captured['rng'] == 123


def test_recorded_sampling_seed_replays_initial_design(monkeypatch):
    from ssp_bayes_opt.bayesian_optimization import BayesianOptimization
    from ssp_bayes_opt import agents

    monkeypatch.setattr(agents, 'SSPAgent', lambda *a, **k: object())

    def draw(seed):
        optimizer = BayesianOptimization(f=lambda x: np.zeros(len(x)),
                                         bounds=np.array([[-5., 5.], [-5., 5.]]),
                                         sampling_seed=seed)
        _, xs, _ = optimizer.initialize_agent(8, 'ssp-hex', ssp_dim=25, rng=seed)
        return xs

    assert np.array_equal(draw(123), draw(123))
    assert not np.array_equal(draw(123), draw(456))


def test_runner_saves_indexed_raw_power_and_window_latency(tmp_path, monkeypatch):
    from experiments import run_agent as runner

    monkeypatch.setattr(runner, '_MEASUREMENTS_DIR', tmp_path)
    monkeypatch.setattr(runner, '_SAMPLE_MEASUREMENTS', [])
    monkeypatch.setattr(runner, '_TRIAL_SEED', 123)
    monkeypatch.setattr(runner, '_POWER_MODE', 'run_only')
    monkeypatch.setattr(runner, '_POWER_OUTPUT', tmp_path / 'legacy')
    capture = SimpleNamespace(data=dict(mode='run_only', error=None, samples=[dict(power_uw=100)], markers=[]))
    windows = [dict(start_timestep=0, end_timestep=2, cycles=10),
               dict(start_timestep=2, end_timestep=3, cycles=20)]
    runner._save_power_capture(capture, windows)
    runner._save_power_capture(capture, windows)
    first = paper.read_json(tmp_path / 'sample-0001.json')
    assert first['fpga']['cycles'] == 30
    assert first['fpga']['latency_ns'] == 150
    assert first['fpga']['windows'] == windows
    assert first['trajectory_index'] == 10
    assert first['seed'] == 123
    assert first['samples'] == capture.data['samples']
    assert paper.read_json(tmp_path / 'sample-0002.json')['trajectory_index'] == 11
