"""Run a single Bayesian optimization trial using pytry.

Supports SSP-hex, SSP-rand, GP, and RFF agents. Optional nengo/loihi/spinnaker
backends for neuromorphic execution. Results are saved as .npz files via pytry.

Usage:
    python run_agent.py --func himmelblau --agent ssp-hex --num-samples 200
"""
import numpy as np
import pytry
import time
try:
    from . import functions
except ImportError:
    import functions

from argparse import ArgumentParser
from contextlib import contextmanager, nullcontext
import tempfile
import os.path
import random
import json
from pathlib import Path
import nengo

import ssp_bayes_opt

try:
    import nengo_loihi
    loihi_lif = nengo_loihi.LoihiLIF()
    loihi_sim = nengo_loihi.Simulator
    loihi_allocator = nengo_loihi.hardware.allocators.Greedy()
except ImportError:
    print('Failed to import nengo_loihi')
    loihi_lif = None
    loihi_sim = None
    loihi_allocator = None
try:
    import nengo_spinnaker
    spinnaker_sim = nengo_spinnaker.Simulator
except ImportError:
    print('Failed to import nengo_spinnaker')
    spinnaker_sim = None

function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133
}

neuron_types = {
    'lif': nengo.LIF(),
    'lifrate': nengo.LIFRate(),
    'loihilif': loihi_lif,
    'direct': nengo.Direct()
}

#'progress_bar':False
sim_types = {
    'cpu': (nengo.Simulator, {'progress_bar':False}),#'gpu': (nengo_ocl.Simulator, {'progress_bar':False}),
    'loihi-sim': (loihi_sim, {'target':'sim','progress_bar':False}),
    'loihi': (loihi_sim,
               {'target':'loihi','precompute':True,
                 'hardware_options':{
                            "snip_max_spikes_per_step": 300,
                            "allocator": loihi_allocator,
                            "n_chips": 15
                        }}
               ),
    'spinnaker': (spinnaker_sim, {}),
}

_POWER_MODE = 'off'
_POWER_OUTPUT = None
_POWER_INTERVAL_MS = 5.0
_POWER_CAPTURE_COUNT = 0
_MEASUREMENTS_DIR = None
_FAILURE_OUTPUT = None
_FPGA_CLOCK_PERIOD_NS = 5.0
_TRIAL_SEED = None
_TRIAL_NUM_INIT_SAMPLES = 10
_SAMPLE_MEASUREMENTS = []


def _save_power_capture(capture, windows=()):
    """Persist raw samples and, for continuous mode, plot annotated power."""
    global _POWER_CAPTURE_COUNT
    _POWER_CAPTURE_COUNT += 1
    base = Path(_POWER_OUTPUT or 'yana-power')
    if _POWER_CAPTURE_COUNT > 1:
        base = base.with_name(f'{base.stem}-{_POWER_CAPTURE_COUNT}{base.suffix}')
    base.parent.mkdir(parents=True, exist_ok=True)
    data = dict(capture.data)
    sample_index = len(_SAMPLE_MEASUREMENTS) + 1
    data.update(seed=_TRIAL_SEED, optimization_sample=sample_index,
                power_interval_ms=_POWER_INTERVAL_MS,
                trajectory_index=_TRIAL_NUM_INIT_SAMPLES + sample_index - 1)
    if windows:
        cycles = sum(window['cycles'] for window in windows)
        data['fpga'] = dict(windows=list(windows), cycles=cycles,
                            clock_period_ns=_FPGA_CLOCK_PERIOD_NS,
                            latency_ns=cycles * _FPGA_CLOCK_PERIOD_NS)
    _SAMPLE_MEASUREMENTS.append(data)
    if _MEASUREMENTS_DIR is not None:
        base = Path(_MEASUREMENTS_DIR) / f'sample-{sample_index:04d}'
        base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix('.json')
    temporary = json_path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
    temporary.replace(json_path)
    print(f'Power data: {json_path}', flush=True)
    samples = data['samples']
    if _POWER_MODE == 'run_only':
        if samples:
            avg_w = sum(row['power_uw'] for row in samples) / len(samples) / 1e6
            print(f'Run-only average power: {avg_w:.6f} W ({len(samples)} samples)', flush=True)
        else:
            print(f'Run-only average power unavailable: no samples ({data.get("error")})', flush=True)
        return
    if not samples:
        print(f'Continuous power plot unavailable: no samples ({data.get("error")})', flush=True)
        return

    import matplotlib.pyplot as plt

    origin = min(row['timestamp_ns'] for row in samples)
    elapsed = [(row['timestamp_ns'] - origin) / 1e9 for row in samples]
    watts = [row['power_uw'] / 1e6 for row in samples]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(elapsed, watts, linewidth=0.9, color='tab:blue')
    ax.set_xlabel('Elapsed time [s] (board monotonic clock)')
    ax.set_ylabel('KR260 SOM power [W]')
    ax.grid(True, alpha=0.3)

    cursor = 0.0
    current_phase = 'idle'
    spans = []
    for marker in data['markers']:
        x = (marker['timestamp_ns'] - origin) / 1e9
        event, phase = marker['event'], marker['phase']
        if event == 'begin':
            if x > cursor:
                spans.append((cursor, x, current_phase))
            cursor, current_phase = x, phase
        elif event == 'end':
            if x > cursor:
                spans.append((cursor, x, phase))
            cursor, current_phase = x, 'idle'
        elif event == 'capture_stop':
            if x > cursor:
                spans.append((cursor, x, current_phase))
    colors = {'idle': '0.85', 'run': 'tab:orange', 'read': 'tab:green',
              'init': 'tab:purple', 'reset': 'tab:gray', 'reset_states': 'tab:pink'}
    for start, end, phase in spans:
        ax.axvspan(start, end, color=colors.get(phase, '0.85'), alpha=0.18)
        if end - start >= 0.03:
            ax.text((start + end) / 2, 0.98, phase, rotation=90,
                    ha='center', va='top', transform=ax.get_xaxis_transform(),
                    fontsize=8)
    fig.tight_layout()
    png_path = base.with_suffix('.png')
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f'Power plot: {png_path}', flush=True)


@contextmanager
def isolated_decoder_cache():
    """Fresh cache per trial, inside the worker's TMPDIR, never the shared cache.

    Keep reuse between sample builds (including NIR export), but discard the
    cache after the trial. A killed trial's TMPDIR is cleared by the orchestrator
    before retrying. Each runner executes trials sequentially in its own process.
    """
    settings = nengo.rc['decoder_cache']
    if not settings.getboolean('enabled'):
        yield None
        return
    previous = {key: settings[key] for key in ('path', 'readonly')}
    with tempfile.TemporaryDirectory(prefix='nengo-decoders-') as directory:
        try:
            settings['path'] = directory
            settings['readonly'] = 'False'
            print(f'Nengo decoder cache: private trial cache at {directory}', flush=True)
            yield Path(directory)
        finally:
            for key, value in previous.items():
                settings[key] = value


class YanaPowerSimulator:
    """Thin Simulator adapter that brackets run() with runtime power capture."""
    def __init__(self, network, **kwargs):
        import nengo_yana
        self._sim = nengo_yana.Simulator(network, **kwargs)
        self._capture = None

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def __enter__(self):
        self._sim.__enter__()
        try:
            if _POWER_MODE != 'off':
                backend = self._sim.device._client.backend
                self._capture = backend.measure_power(
                    mode=_POWER_MODE, interval_ms=_POWER_INTERVAL_MS)
                self._capture.__enter__()
        except BaseException:
            self._sim.close()
            raise
        return self

    def run(self, *args, **kwargs):
        return self._sim.run(*args, **kwargs)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._capture is not None:
                self._capture.__exit__(exc_type, exc, tb)
        finally:
            # Release FIFO ownership before local JSON/plot processing.
            self._sim.__exit__(exc_type, exc, tb)
        if self._capture is not None:
            _save_power_capture(self._capture, self._sim.device.run_windows)
        return False


def configure_yana_backend(device_ip='kria.local', readout_every=2500,
                           deployment_dir=None):
    """Register YANA, using a host for FPGA or ``None`` for its emulator."""
    sim_types['yana'] = (YanaPowerSimulator, {
        'progress_bar': False,
        'device_ip': device_ip,
        'readout_every': readout_every,
        'validate_neuron_params': True,
        'deployment_dir': deployment_dir,
    })
    return sim_types['yana']


def get_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--func', dest='function_name', type=str, default='branin-hoo')
    parser.add_argument('--agent', dest='agent_type', type=str, default='ssp-hex')
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float, default=1.)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.0)
    parser.add_argument('--decay', action='store_true')

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=97)
    parser.add_argument('--n-scales', dest='n_scales', type=int, default=-1)
    parser.add_argument('--n-rotates', dest='n_rotates', type=int, default=-1)
    parser.add_argument('--len-scale', dest='len_scale', type=float, default=-1)

    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None,
                        help='reproducible trial seed (default: random)')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')

    parser.add_argument('--nengo', action='store_true')
    parser.add_argument('--backend', dest='backend', type=str, default="cpu",
                        help='simulator backend: cpu, yana (FPGA), or yana-emulator')
    parser.add_argument('--num-neurons', dest='num_neurons', type=int,
                        default=10)  # 7 is max for spinnaker with d=97, 8 for loihi
    parser.add_argument('--encoder-sparsity', type=float, default=0.0)
    parser.add_argument('--yana-deployment-dir', type=Path,
                        help='dedicated reusable deployment workspace; one per worker')
    parser.add_argument('--measurements-dir', type=Path,
                        help='write indexed per-sample power and FPGA timing JSON')
    parser.add_argument('--failure-output', type=Path,
                        help='write structured failure metadata for the orchestrator')
    parser.add_argument('--fpga-clock-period-ns', type=float, default=5.0,
                        help='deployed FPGA clock period, not simulation dt (default: 5 ns)')
    parser.add_argument('--yana-host', default='kria.local',
                        help='KR260 host for the YANA backend')
    parser.add_argument('--yana-readout-every', type=int, default=2500,
                        help='YANA readout window in timesteps (default: 2500)')
    parser.add_argument('--power-mode', choices=('off', 'run_only', 'continuous'),
                        default='off', help='opt-in YANA power capture mode')
    parser.add_argument('--power-interval-ms', type=float, default=5.0)
    parser.add_argument('--power-output', type=Path,
                        help='base path for power JSON/plot output')
    args = parser.parse_args(argv)
    if not 0.0 <= args.encoder_sparsity <= 1.0:
        parser.error('--encoder-sparsity must be in [0, 1]')
    if args.num_samples < 1 or args.num_trials < 1 or args.num_neurons < 1:
        parser.error('sample, trial and neuron counts must be positive')
    if not np.isfinite(args.fpga_clock_period_ns) or args.fpga_clock_period_ns <= 0:
        parser.error('--fpga-clock-period-ns must be finite and positive')
    return args

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('agent_type', agent_type='ssp-hex')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('ssp length scale', len_scale=4)
        self.param('UCB Beta', beta_ucb=1.0)
        self.param('MI gamma_c', gamma=0.0)
        self.param('ssp dim', ssp_dim=151)
        self.param('n_scales', n_scales=4)
        self.param('n_rotates', n_rotates=4)
        self.param('trial number', trial_num=None)

        self.param('use nengo', nengo=False)
        self.param('nengo backend', backend='cpu')
        self.param('num neurons', num_neurons=8)
        self.param('encoder sparsity', encoder_sparsity=0.0)
        self.param('sim time', sim_time=2.5)

        self.param('use beta decay', decay=False)
    
    def evaluate(self, p):
        global _TRIAL_NUM_INIT_SAMPLES
        _TRIAL_NUM_INIT_SAMPLES = p.num_init_samples
        target, pbounds, _ = functions.factory(p.function_name)
        #target, pbounds = functions.rescale(target,pbounds)
        budget = p.num_samples
        if p.decay:
            var_decay = -p.beta_ucb / budget
        else:
            var_decay = 0
        # was 0 before

        if p.nengo:
            sim_time = p.sim_time
            neuron_type = neuron_types['loihilif'] if 'loihi' in p.backend else neuron_types['lif']
            sim_type, sim_args = sim_types[p.backend]

            optimizer = ssp_bayes_opt.NengoBayesianOptimization(f=target,
                                                           bounds=pbounds,
                                                           verbose=p.verbose,
                                                           random_state=p.seed,
                                                           sampling_seed=p.seed)
            start = time.thread_time_ns()
            optimizer.maximize(init_points=p.num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type=p.agent_type,
                               ssp_dim=p.ssp_dim,
                               n_scales=p.n_scales,
                               n_rotates=p.n_rotates,
                               length_scale=p.len_scale,
                               decoder_method='direct-optim',
                               gamma_c=p.gamma,
                               # decoder_method='network-optim',
                               beta_ucb=p.beta_ucb,
                               var_decay=var_decay,
                               neurons_per_dim=p.num_neurons,
                               encoder_sparsity=p.encoder_sparsity,
                               rng=p.seed,
                               neuron_type=neuron_type,
                               sim_type=sim_type, sim_args=sim_args,
                               sim_time=sim_time, tau=0.05,
                               save_memory=False,
                               partitions=None
                               )
            elapsed_time = time.thread_time_ns() - start
            sim_times = optimizer.sim_times
        else:
            optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                                       bounds=pbounds, 
                                                       verbose=p.verbose,

                                                       sampling_seed=p.seed)
            start = time.thread_time_ns()
            optimizer.maximize(init_points=p.num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type=p.agent_type,
                               ssp_dim=p.ssp_dim,
                               n_scales=p.n_scales,
                               n_rotates=p.n_rotates,
                               length_scale=p.len_scale,
                               decoder_method='direct-optim',
                               gamma_c=p.gamma,
                               # decoder_method='network-optim',
                               beta_ucb=p.beta_ucb,
                               var_decay=var_decay,
                               rand_pad=False,use_jac=True
                               )
            elapsed_time = time.thread_time_ns() - start
            sim_times = None
        

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        if "styblinski-tang" in p.function_name:
            true_max_val = 39.16599*pbounds.shape[0]
        elif "rastrigin" in p.function_name:
            true_max_val = 0
        elif "rosenbrock " in p.function_name:
            true_max_val = 0
        else:
            true_max_val = function_maximum_value[p.function_name] 
        regrets = true_max_val - vals
        cum_reg = np.divide(np.cumsum(regrets), np.arange(1,p.num_init_samples+budget+1))
        print(optimizer.max)
        print(cum_reg[-1])
        print(np.mean(optimizer.times) * 1e-9)
        
        return dict(
            regret=regrets,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            times = optimizer.times,#selected_len_scale = optimizer.length_scale,
            full_times= optimizer.full_times,
            sim_times=sim_times,
            effective_ssp_dim=getattr(optimizer.agt, 'ssp_dim', None),
            fpga_cycles=np.asarray([m['fpga']['cycles'] for m in _SAMPLE_MEASUREMENTS if 'fpga' in m]),
            fpga_latencies_ns=np.asarray([m['fpga']['latency_ns'] for m in _SAMPLE_MEASUREMENTS if 'fpga' in m]),
            memory = optimizer.memory,
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
            total_time=optimizer.total_time,
            params=p
        )



def main(argv=None):
    global _POWER_MODE, _POWER_INTERVAL_MS, _POWER_OUTPUT, _MEASUREMENTS_DIR
    global _FAILURE_OUTPUT, _FPGA_CLOCK_PERIOD_NS, _TRIAL_SEED, _SAMPLE_MEASUREMENTS
    args = get_args(argv)
    _FAILURE_OUTPUT = args.failure_output
    _FPGA_CLOCK_PERIOD_NS = args.fpga_clock_period_ns
    if args.backend in ('yana', 'yana-emulator'):
        if not args.nengo:
            raise SystemExit(f'--backend {args.backend} requires --nengo')
        device_ip = args.yana_host if args.backend == 'yana' else None
        sim_types[args.backend] = configure_yana_backend(
            device_ip, args.yana_readout_every, args.yana_deployment_dir)
    if args.power_mode != 'off' and args.backend != 'yana':
        raise SystemExit('--power-mode requires --backend yana')
    _POWER_MODE = args.power_mode
    _POWER_INTERVAL_MS = args.power_interval_ms
    _POWER_OUTPUT = args.power_output or Path('data') / 'power' / f'yana-{args.power_mode}'
    # args.nengo = True

    # random.seed(1)
    seeds = (
        [args.seed + i for i in range(args.num_trials)]
        if args.seed is not None
        else [random.SystemRandom().randrange(1, 2**31 - 1) for _ in range(args.num_trials)]
    )

    if args.nengo:
        data_path = os.path.join(os.getcwd(),
                                 args.data_dir, args.function_name, args.agent_type + '_nengo-' + args.backend)
    else:
        data_path = os.path.join(os.getcwd(),
                             args.data_dir,args.function_name,args.agent_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for trial_index, seed in enumerate(seeds, 1):
        _TRIAL_SEED = seed
        _SAMPLE_MEASUREMENTS = []
        _MEASUREMENTS_DIR = args.measurements_dir
        if _MEASUREMENTS_DIR is not None and args.num_trials > 1:
            _MEASUREMENTS_DIR = _MEASUREMENTS_DIR / f'trial-{trial_index:04d}'
        params = {'function_name':args.function_name,
                  'agent_type':args.agent_type,
                  'num_samples':args.num_samples,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'verbose':False,
                  'ssp_dim':args.ssp_dim,
                  'len_scale':args.len_scale,
                  'beta_ucb':args.beta_ucb,
                  'gamma':args.gamma,
                  'nengo':args.nengo,
                  'backend':args.backend,
                  'n_scales':args.n_scales,
                  'n_rotates':args.n_rotates,
                  'decay': args.decay,
                  'num_neurons': args.num_neurons,
                  'encoder_sparsity': args.encoder_sparsity
                  }
        cache = isolated_decoder_cache() if args.backend in ('yana', 'yana-emulator') else nullcontext()
        with cache:
            r = SamplingTrial().run(**params)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        try:
            from yana.deploy.accelerator import HardwareCapacityError
            capacity = isinstance(exc, HardwareCapacityError)
        except ImportError:
            capacity = False
        failure = dict(kind='capacity' if capacity else 'execution',
                       exception_type=f'{type(exc).__module__}.{type(exc).__name__}',
                       message=str(exc), traceback=traceback.format_exc(), seed=_TRIAL_SEED,
                       captured_samples=len(_SAMPLE_MEASUREMENTS),
                       failed_sample=(len(_SAMPLE_MEASUREMENTS) + 1) if capacity else None,
                       violations=getattr(exc, 'violations', []))
        if _FAILURE_OUTPUT is not None:
            _FAILURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            temporary = _FAILURE_OUTPUT.with_suffix('.tmp')
            temporary.write_text(json.dumps(failure, indent=2))
            temporary.replace(_FAILURE_OUTPUT)
        traceback.print_exc()
        raise SystemExit(2 if capacity else 1)
