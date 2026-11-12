"""Continue completed ResNet50 perception with one fusion and one joint epoch.

Use exactly one visible GPU. Preserve all source experiments and run the stages
sequentially, saving exit codes and an automatic comparison on the same split.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def stage_result(directory):
    manifest = json.loads((directory/'config.json').read_text())
    epochs = [row for line in (directory/'metrics.jsonl').read_text().splitlines()
              if 'validation' in (row := json.loads(line))]
    best = min(epochs, key=lambda row:row['validation']['final']['ade'])
    return dict(directory=str(directory), best_checkpoint=str(directory/'best.pt'),
                best_epoch=best['epoch'], completed_epochs=len(epochs),
                completed_steps=epochs[-1]['step'], batch_size=manifest['args']['batch_size'],
                image_size=manifest['image_size'], selection_sha256=manifest['selection_sha256'],
                split_sha256=manifest['split_sha256'],
                cuda_visible_devices=manifest['cuda_visible_devices'],
                cuda_device_count=manifest['cuda_device_count'], validation=best['validation'])


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'runs/city_phase1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(root/'resnet50_staged_v2'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min-free-memory-mib', type=int, default=8500,
                        help='wait before each stage until the selected GPU has this much free memory')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.min_free_memory_mib < 1:
        raise ValueError('positive batch size/free memory and nonnegative worker count required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise ValueError('set CUDA_VISIBLE_DEVICES to exactly one physical GPU UUID')
    perception = root/'perception_resnet50_ce_dice_v1'
    ego = root/'ego_full_cpu'
    for source in (perception, ego):
        if int(source.with_suffix('.exit_code').read_text().strip()) != 0:
            raise RuntimeError(f'source experiment failed: {source}')
        if not (source/'best.pt').is_file():
            raise RuntimeError(f'source checkpoint missing: {source}')
    source_manifest = json.loads((perception/'config.json').read_text())
    saved = source_manifest['args']
    selection = Path(saved['selection_file'])
    split_path = Path(saved['split_file']) if saved['split_file'] else project/'configs/nuscenes_splits.json'
    for key,path in (('selection_sha256',selection), ('split_sha256',split_path)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_manifest[key]:
            raise ValueError(f'perception source data changed: {key}')
    baseline_directories = (ego, root/'fusion_full_gpu1', root/'joint_full_gpu1')
    baselines = {path.name:stage_result(path) for path in baseline_directories}
    for baseline in baselines.values():
        for key in ('selection_sha256', 'split_sha256'):
            if baseline[key] != source_manifest[key]:
                raise ValueError(f'baseline uses different data: {key}')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-u', '-m', 'scripts.train_staged_driving',
              '--root', saved['root'], '--version', saved['version'],
              '--selection-file', str(selection), '--model-config', saved['model_config'],
              '--image-height', str(saved['image_height']), '--image-width', str(saved['image_width']),
              '--batch-size', str(args.batch_size), '--epochs', '1',
              '--workers', str(args.workers), '--threads', '4', '--device', 'cuda',
              '--seed', str(saved['seed']), '--bev-loss', saved['bev_loss'],
              '--lr', str(saved['lr']), '--encoder-lr-scale', str(saved['encoder_lr_scale']),
              '--backbone-lr-scale', str(saved['backbone_lr_scale']),
              '--ego-dropout', str(saved['ego_dropout']), '--visual-weight', str(saved['visual_weight'])]
    if saved['split_file']:
        common += ['--split-file', saved['split_file']]
    write_json(output/'plan.json', dict(cuda_visible_devices=visible, batch_size=args.batch_size,
        min_free_memory_mib=args.min_free_memory_mib,
        sizes=source_manifest['sizes'], perception_checkpoint=str(perception/'best.pt'),
        ego_checkpoint=str(ego/'best.pt'), epochs={'fusion':1, 'joint':1},
        baselines=baselines, notes=[
            'Same data selection and validation split as the old Tiny pipeline.',
            'Image backbone, perception pretraining, image size and loss differ from Tiny.',
            'If batch size differs, optimizer update counts also differ; this is not an equal-compute ablation.']))
    log = (output/'pipeline.log').open('x')

    def record(phase, **values):
        row = dict(phase=phase, time_utc=datetime.now(timezone.utc).isoformat(),
                   pid=os.getpid(), cuda_visible_devices=visible, **values)
        log.write(json.dumps(row)+'\n')
        log.flush()
        write_json(output/'status.json', row)
        print(json.dumps(row), flush=True)

    try:
        for stage in ('fusion', 'joint'):
            while True:
                free_memory = int(subprocess.check_output([
                    'nvidia-smi', '-i', visible, '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free_memory >= args.min_free_memory_mib:
                    break
                record('waiting_for_gpu_memory', stage=stage, free_memory_mib=free_memory,
                       required_memory_mib=args.min_free_memory_mib)
                time.sleep(30)
            flags = (['--perception-checkpoint', str(perception/'best.pt'),
                      '--ego-checkpoint', str(ego/'best.pt')] if stage == 'fusion' else
                     ['--init-checkpoint', str(output/'fusion/best.pt')])
            command = common+['--stage', stage]+flags+['--output', str(output/stage)]
            write_json(output/(stage+'.command.json'), command)
            record('training', stage=stage, command=command)
            with (output/(stage+'.log')).open('x') as child_log:
                result = subprocess.run(command, cwd=project, stdout=child_log, stderr=subprocess.STDOUT)
            (output/(stage+'.exit_code')).write_text(str(result.returncode)+'\n')
            if result.returncode:
                raise RuntimeError(f'{stage} failed with exit code {result.returncode}; inspect {stage}.log')
            if not (output/stage/'best.pt').is_file():
                raise RuntimeError(f'{stage} completed without best.pt')
            record('stage_finished', stage=stage)
        jobs = {stage:stage_result(output/stage) for stage in ('fusion', 'joint')}
        comparisons = {}
        for stage in ('fusion', 'joint'):
            current = jobs[stage]
            for key in ('selection_sha256', 'split_sha256'):
                if current[key] != source_manifest[key]:
                    raise ValueError(f'{stage} data mismatch: {key}')
            previous = baselines[stage+'_full_gpu1']['validation']['final']
            new = current['validation']['final']
            comparisons[stage] = dict(
                ade_old_m=previous['ade'], ade_new_m=new['ade'],
                ade_improvement_m=previous['ade']-new['ade'],
                ade_improvement_percent=100*(previous['ade']-new['ade'])/previous['ade'],
                fde_old_m=previous['fde'], fde_new_m=new['fde'],
                ade_improvement_vs_ego_m=baselines['ego_full_cpu']['validation']['final']['ade']-new['ade'])
        write_json(output/'comparison.json', dict(sizes=source_manifest['sizes'], jobs=jobs,
            baselines=baselines, comparisons=comparisons,
            interpretation='Positive ADE improvement means lower error. Combined engineering comparison, not isolated causal or equal-compute evidence.'))
        record('complete', checkpoint=str(output/'joint/best.pt'), comparison=str(output/'comparison.json'),
               comparisons=comparisons)
    except BaseException as error:
        record('failed', error=str(error))
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
