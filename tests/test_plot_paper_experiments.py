"""Read-only snapshot plots: correct metrics, shard merging and write isolation."""

import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments import plot_paper_experiments as plot
from experiments import run_paper_experiments as paper
from test_paper_experiments import fake_artifacts


@pytest.fixture
def config():
    return paper.sweep_configs(num_samples=2)[0]


def make_trial(root, config, index=1):
    parent = paper.config_path(root, config)
    parent.mkdir(parents=True, exist_ok=True)
    paper.ensure_config(parent, config)
    trial = parent / f'trial-{index:04d}'
    seed = paper.prepare_trial(trial, config, index, 'fake')
    fake_artifacts(trial, config, seed)
    paper.mark_complete(trial, config, seed)
    return trial


def update_npz(path, **updates):
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    values.update(updates)
    np.savez(path, **values)


def recommit(trial, config):
    paper.mark_complete(trial, config, paper.read_json(trial / 'trial.json')['seed'])


def file_snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_energy_is_mean_of_power_times_fpga_latency_per_sample(tmp_path, config):
    trial = make_trial(tmp_path, config)
    first, second = [trial / 'measurements' / f'sample-{i:04d}.json' for i in (1, 2)]
    data = paper.read_json(first)
    data['samples'] = [dict(phase='run', timestamp_ns=100, power_uw=1_000_000)]
    paper.atomic_json(first, data)
    data = paper.read_json(second)
    data['samples'] = [dict(phase='run', timestamp_ns=i, power_uw=3_000_000) for i in (99, 100, 101)]
    paper.atomic_json(second, data)
    trajectory = next((trial / 'trajectory').glob('*.npz'))
    update_npz(trajectory, regret=np.arange(1., 13.))
    recommit(trial, config)
    metrics = plot.read_trial(trial / 'complete.json')
    # 100 and 200 cycles at 5 ns; sample powers 1 W and 3 W.
    assert metrics.mean_energy_j == pytest.approx((1 * 500e-9 + 3 * 1000e-9) / 2)
    # Includes all 10 initial observations, as in the runner's cumulative average.
    assert metrics.final_average_regret == pytest.approx(6.5)


def test_only_completed_trials_are_read_not_active_or_archived(tmp_path, config, monkeypatch):
    complete = make_trial(tmp_path, config)
    active = paper.config_path(tmp_path, config) / 'trial-0002'
    active.mkdir()
    (active / 'in-progress.npz').write_bytes(b'half-written')
    archive = complete.with_name('trial-0001-attempts') / 'attempt-0001'
    shutil.copytree(complete, archive)
    before = file_snapshot(tmp_path)
    original = Path.open

    def protected_open(path, mode='r', *args, **kwargs):
        if plot.inside(path.resolve(), tmp_path.resolve()):
            assert not any(ch in mode for ch in 'wax+'), 'reader tried to write source data'
            assert not plot.inside(path.resolve(), active.resolve()), 'reader opened active trial'
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', protected_open)
    metrics = plot.load_trials([tmp_path])
    assert len(metrics) == 1
    assert metrics[0].path == complete
    monkeypatch.setattr(Path, 'open', original)
    assert file_snapshot(tmp_path) == before
    assert not list(tmp_path.rglob('.lock'))


def test_merges_two_shards_and_deduplicates_copies(tmp_path, config):
    one, two = tmp_path / 'one', tmp_path / 'two'
    make_trial(one, config)
    make_trial(two, dict(config, num_neurons=10))
    assert len(plot.load_trials([one, two])) == 2
    copy = tmp_path / 'copy'
    shutil.copytree(one, copy)
    assert len(plot.load_trials([one, two, copy])) == 2


def test_conflicting_trial_copies_are_rejected(tmp_path, config):
    one, two = tmp_path / 'one', tmp_path / 'two'
    source = make_trial(one, config)
    shutil.copytree(one, two)
    trial = paper.config_path(two, config) / source.name
    path = trial / 'measurements' / 'sample-0001.json'
    data = paper.read_json(path)
    data['samples'][0]['power_uw'] += 1
    paper.atomic_json(path, data)
    recommit(trial, config)
    with pytest.raises(ValueError, match='Conflicting copies'):
        plot.load_trials([one, two])


@pytest.mark.parametrize('damage', ['checksum', 'empty-power', 'wrong-timing', 'missing-file'])
def test_invalid_completed_trials_are_skipped_without_repair(tmp_path, config, damage):
    trial = make_trial(tmp_path, config)
    measurement = trial / 'measurements' / 'sample-0001.json'
    if damage == 'missing-file':
        measurement.unlink()
    else:
        data = paper.read_json(measurement)
        if damage == 'empty-power':
            data['samples'] = []
        elif damage == 'wrong-timing':
            data['fpga']['latency_ns'] += 1
        else:
            data['samples'][0]['power_uw'] += 1
        paper.atomic_json(measurement, data)
        if damage != 'checksum':
            saved = paper.read_json(trial / 'complete.json')
            saved['files']['measurements/sample-0001.json'] = paper.checksum(measurement)
            paper.atomic_json(trial / 'complete.json', saved)
    before = file_snapshot(tmp_path)
    warnings = []
    assert plot.load_trials([tmp_path], warnings.append) == []
    assert len(warnings) == 1
    assert file_snapshot(tmp_path) == before


def test_manifest_cannot_read_outside_trial_tree(tmp_path, config):
    trial = make_trial(tmp_path, config)
    saved = paper.read_json(trial / 'complete.json')
    outside = tmp_path / 'outside.npz'
    outside.write_bytes(b'not an input')
    key = 'trajectory/../../../outside.npz'
    trajectory = next(key for key in saved['files'] if key.endswith('.npz'))
    saved['files'][key] = saved['files'].pop(trajectory)
    paper.atomic_json(trial / 'complete.json', saved)
    with pytest.raises(ValueError, match='Unsafe manifest path'):
        plot.read_trial(trial / 'complete.json')


def test_mixed_budgets_are_not_combined(tmp_path, config):
    make_trial(tmp_path / 'one', config)
    make_trial(tmp_path / 'two', dict(config, num_neurons=10, num_samples=3))
    trials = plot.load_trials([tmp_path / 'one', tmp_path / 'two'])
    with pytest.raises(ValueError, match='different fixed settings'):
        plot.aggregate(trials)


def test_averages_trials_to_one_point_and_records_available_count(tmp_path, config):
    make_trial(tmp_path, config, index=1)
    other = make_trial(tmp_path, config, index=2)
    update_npz(next((other / 'trajectory').glob('*.npz')), regret=np.full(12, 3.))
    recommit(other, config)
    rows = plot.aggregate(plot.load_trials([tmp_path]))
    assert len(rows) == 1
    assert rows[0]['completed_trials'] == 2
    assert rows[0]['final_cumulative_average_regret'] == 2


def test_output_cannot_be_inside_any_input_or_symlink_to_one(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    with pytest.raises(ValueError, match='outside every input'):
        plot.output_directory(root / 'plots', [root])
    link = tmp_path / 'linked-output'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='outside every input'):
        plot.output_directory(link, [root])
    outside = tmp_path / 'output'
    outside.mkdir()
    (outside / plot.OUTPUTS[0]).symlink_to(root / 'data')
    with pytest.raises(ValueError, match='points into an input'):
        plot.output_directory(outside, [root])


def test_cli_generates_plots_and_csv_without_touching_inputs(tmp_path, config, monkeypatch):
    source, output = tmp_path / 'input', tmp_path / 'output'
    trial = make_trial(source, config)
    before = file_snapshot(source)
    original_open, original_mkdir = Path.open, Path.mkdir

    def guarded_open(path, mode='r', *args, **kwargs):
        if plot.inside(path.resolve(), source.resolve()):
            assert not any(ch in mode for ch in 'wax+')
        return original_open(path, mode, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):
        assert not plot.inside(path.resolve(), source.resolve()), 'writer tried to create input directory'
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(Path, 'mkdir', guarded_mkdir)
    # Even an output hardlink must not cause the original input file to be overwritten.
    output.mkdir()
    (output / plot.OUTPUTS[0]).hardlink_to(trial.parent / 'config.json')
    monkeypatch.setenv('MPLCONFIGDIR', str(source / 'must-not-be-created'))
    plot.main(['--artifact-dirs', str(source), '--output-dir', str(output)])
    assert file_snapshot(source) == before
    for name in plot.OUTPUTS[:2]:
        assert (output / name).read_bytes().startswith(b'\x89PNG')
    assert 'mean_inference_energy_j' in (output / 'summary.csv').read_text()
    assert not list(output.glob('.plot-*'))
    assert not list(output.glob('.matplotlib-*'))
    assert not (source / 'must-not-be-created').exists()


def test_lines_follow_neuron_count_not_energy_order(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    original = plt.subplots
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(plt, 'subplots', capture)
    rows = [dict(function_name='himmelblau', ssp_dim=25, encoder_sparsity=0.0,
                 num_neurons=n, completed_trials=2, mean_inference_energy_j=e,
                 final_cumulative_average_regret=r) for n, e, r in [(10, .001, 1), (5, .002, 2)]]
    plot.draw_plot(rows, 'dimensions', tmp_path / 'plot.png')
    ax = captured[0][1][0, 0]
    assert list(ax.lines[0].get_xdata()) == [2, 1]
    assert list(ax.lines[0].get_ydata()) == [2, 1]
    assert [text.get_text() for text in ax.texts] == ['5 (2)', '10 (2)']


def test_empty_snapshot_still_produces_explanatory_figures(tmp_path):
    source, output = tmp_path / 'input', tmp_path / 'output'
    source.mkdir()
    plot.main(['--artifact-dirs', str(source), '--output-dir', str(output)])
    assert all((output / name).exists() for name in plot.OUTPUTS)
    assert len((output / 'summary.csv').read_text().splitlines()) == 1
