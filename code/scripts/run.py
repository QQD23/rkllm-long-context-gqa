"""One hash-checked, mutually exclusive model run; no frequency writes or ADB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from verify_runtime import ROOT, verify


def clean_environment(runtime, mode, patch):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('RKDUANYADIAN', 'WP_')) or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT'):
            env.pop(key)
    env['LD_LIBRARY_PATH'] = str(runtime)
    env['WP_IGNORE_EOS'] = '1'
    env['WP_PROGRESS_EVERY'] = '32'
    if mode != 'baseline':
        env['LD_PRELOAD'] = str(patch)
        env['RKDUANYADIAN_MODE'] = mode
    return env


def temperatures():
    values = {p.parent.name: int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    if not values:
        raise RuntimeError('Thermal sensors unavailable')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'shadow', 'tile'], required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input-ids', type=Path, required=True)
    parser.add_argument('--output-tokens', type=int, default=2)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        parser.error('Requires Linux aarch64 / RK3588')
    if 'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes().decode(errors='replace'):
        parser.error('Only RK3588 is supported')
    runtime = args.runtime_dir.resolve()
    hashes = verify(runtime)
    ids = [int(v) for v in args.input_ids.read_text().split()]
    if not ids or min(ids) < 0 or max(ids) > 2147483647 or not 2 <= args.output_tokens <= 65536 or len(ids) + args.output_tokens > args.context or args.timeout <= 0:
        parser.error('Invalid token IDs, lengths, context, or timeout')
    worker = ROOT / 'build/position_worker'
    patch = ROOT / 'build/libpv_gqa_v23.so'
    if not args.model.is_file() or not worker.is_file() or (args.mode != 'baseline' and not patch.is_file()):
        parser.error('Missing model or built executable/patch')
    import fcntl
    lock = (ROOT / 'build/run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # This lock coordinates this package only. Users must stop unrelated workloads.
    start_temp = temperatures()
    if max(start_temp.values()) > 55000:
        raise RuntimeError('Above 55 C: wait for cooling, then retry; model not started')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [str(worker), str(args.model.resolve()), str(args.input_ids.resolve()), str(out), str(args.output_tokens), str(args.context), '240', '1', 'tokens', '1.3.0']
    record = dict(mode=args.mode, runtime_sha256=hashes, input_tokens=len(ids), output_tokens_requested=args.output_tokens,
                  input_sha256=hashlib.sha256(args.input_ids.read_bytes()).hexdigest(), start_temperature_before_loading_mC=start_temp,
                  cpu_mask=240, frequency_writes=False, state='running', command=command)
    if patch.is_file():
        record['patch_sha256'] = hashlib.sha256(patch.read_bytes()).hexdigest()
    def save():
        (out / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    save()
    child = None
    try:
        with (out / 'worker.log').open('x') as log, (out / 'temperature.jsonl').open('x') as thermal:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                     env=clean_environment(runtime, args.mode, patch), text=True)
            child.stdin.write('long events\nquit\n')
            child.stdin.close()
            begin = time.monotonic()
            while child.poll() is None:
                ts = temperatures()
                thermal.write(json.dumps(dict(monotonic_s=time.monotonic(), sensors_mC=ts)) + '\n')
                thermal.flush()
                if max(ts.values()) >= 85000:
                    raise RuntimeError('85 C thermal limit; this run is incomplete')
                if time.monotonic() - begin > args.timeout:
                    raise TimeoutError('Run timed out')
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError('Worker failed; inspect worker.log')
        result = json.loads((out / 'long.json').read_text())
        callbacks = [x for x in result['callbacks'] if x['state'] in (0, 1) and x['has_result'] and x['token_id'] >= 0]
        if not result['valid'] or len(callbacks) != args.output_tokens or result['perf']['prefill_tokens'] != len(ids):
            raise RuntimeError('Output/input count or worker validity check failed')
        record.update(state='complete', actual_output_tokens=len(callbacks), perf=result['perf'])
    except BaseException as exc:
        record.update(state='failed', error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        save()
        lock.close()


if __name__ == '__main__':
    main()
