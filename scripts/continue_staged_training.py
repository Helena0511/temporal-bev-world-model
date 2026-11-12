"""Wait for independent pretraining, then run fusion/joint on the same one GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'runs/city_phase1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise ValueError('set CUDA_VISIBLE_DEVICES to one physical GPU or UUID')
    status = root/'staged_full_pipeline_status.json'
    if status.exists():
        raise ValueError('a pipeline status file already exists; inspect it before restarting')
    log = (root/'staged_full_pipeline.log').open('x')

    def record(phase, **kwargs):
        row = dict(phase=phase, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   cuda_visible_devices=visible, **kwargs)
        log.write(json.dumps(row)+'\n')
        log.flush()
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(status)
        print(json.dumps(row), flush=True)

    try:
        record('waiting_for_pretraining', jobs=['perception_full_gpu1', 'ego_full_cpu'])
        for name in ('perception_full_gpu1', 'ego_full_cpu'):
            exit_path = root/(name+'.exit_code')
            while not exit_path.exists():
                time.sleep(10)
            code = int(exit_path.read_text().strip())
            if code != 0:
                raise RuntimeError(f'{name} failed with exit code {code}; inspect {name}.log')
            if not (root/name/'best.pt').is_file():
                raise RuntimeError(f'{name} completed without best.pt')
        common = [sys.executable, '-m', 'scripts.train_staged_driving', '--selection-file',
                  str(root/'full_selection.json'), '--model-config',
                  str(project/'configs/bev_attention_city60_pilot.json'),
                  '--image-height', '128', '--image-width', '352', '--batch-size', '4',
                  '--epochs', '1', '--workers', '4', '--threads', '4', '--device', 'cuda']
        experiments = [
            ('fusion_full_gpu1', ['--stage', 'fusion', '--perception-checkpoint',
                str(root/'perception_full_gpu1/best.pt'), '--ego-checkpoint', str(root/'ego_full_cpu/best.pt')]),
            ('joint_full_gpu1', ['--stage', 'joint', '--init-checkpoint', str(root/'fusion_full_gpu1/best.pt')]),
        ]
        for name, flags in experiments:
            output = root/name
            child_log = root/(name+'.log')
            if output.exists() or child_log.exists():
                raise RuntimeError(f'preserving existing experiment: {name}')
            command = common+flags+['--output', str(output)]
            (root/(name+'.command.json')).write_text(json.dumps(command, indent=2))
            record('training', job=name, command=command)
            with child_log.open('x') as file:
                result = subprocess.run(command, cwd=project, stdout=file, stderr=subprocess.STDOUT)
            (root/(name+'.exit_code')).write_text(str(result.returncode)+'\n')
            if result.returncode:
                raise RuntimeError(f'{name} failed with exit code {result.returncode}; inspect {child_log.name}')
            record('stage_finished', job=name)
        record('complete', checkpoint=str(root/'joint_full_gpu1/best.pt'))
    except BaseException as error:
        record('failed', error=str(error))
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
