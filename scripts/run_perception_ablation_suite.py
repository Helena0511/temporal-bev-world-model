"""Run registered perception improvements sequentially on one physical GPU.

Wait for the previous joint job, freeze Python sources, smoke every candidate,
run matched development experiments, evaluate their frozen best checkpoints on
official validation, then retrain up to two development winners and a matched
baseline on the full city-condition training set. Preserve every failed attempt.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def training_result(path):
    epochs=[r for line in (path/'metrics.jsonl').read_text().splitlines()
            if 'validation' in (r:=json.loads(line))]
    best=max(epochs,key=lambda r:r['validation']['foreground_miou'])
    if not (path/'best.pt').is_file(): raise ValueError('completed trial has no best checkpoint')
    return dict(directory=str(path),checkpoint=str(path/'best.pt'),best_epoch=best['epoch'],
        completed_epochs=len(epochs),updates=epochs[-1]['step'],validation=best['validation'],
        max_peak_allocated_mib=max(r['cuda_peak_allocated_mib'] or 0 for r in epochs))


def qualifies(candidate, baseline, rule):
    c,b=candidate['validation'],baseline['validation']
    return (c['foreground_miou']>b['foreground_miou']+rule['min_foreground_miou_gain']
        and c['classes']['human']['iou']>b['classes']['human']['iou']+rule['min_human_iou_gain']
        and c['classes']['vehicle']['iou']>=b['classes']['vehicle']['iou']-rule['max_vehicle_iou_drop']
        and (c['classes']['human']['precision'] or 0)>=(b['classes']['human']['precision'] or 0)-rule['max_human_precision_drop'])


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root',required=True)
    parser.add_argument('--min-free-memory-mib',type=int,default=11000)
    args=parser.parse_args()
    root=Path(args.suite_root).resolve(); plan_path=root/'plan.json'
    plan=json.loads(plan_path.read_text()); plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible.startswith('GPU-') or ',' in visible:
        raise ValueError('set one physical GPU UUID in CUDA_VISIBLE_DEVICES')
    lock=root/'runner.lock'
    with lock.open('x') as handle: handle.write(str(os.getpid())+'\n')
    log=(root/'pipeline.log').open('x')
    def record(phase,**values):
        row=dict(phase=phase,time_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(),cuda_visible_devices=visible,**values)
        log.write(json.dumps(row)+'\n'); log.flush(); write_json(root/'status.json',row)
        print(json.dumps(row),flush=True)
    results=dict(screen={},official_screen={},full={},smoke={},promotions=[])
    def refresh():
        write_json(root/'comparison.json',results)
        with (root/'comparison.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['phase','variant','best_epoch','updates','vehicle_iou','human_iou','foreground_miou',
                'human_precision','human_recall','supported_visible_20_40m_human_center_2m_coverage'])
            for phase in ('screen','official_screen','full'):
                for name,row in results[phase].items():
                    v=row['validation']; h=v['classes']['human']
                    writer.writerow([phase,name,row.get('best_epoch'),row.get('updates'),v['classes']['vehicle']['iou'],
                        h['iou'],v['foreground_miou'],h['precision'],h['recall'],
                        v['object_center_coverage']['supported_visible_20-40m']['human']['coverage']])
        lines=['# 感知逐项改进对照','',
            '每个screen变体仅修改一项；2,048训练序列/256开发序列来自互斥的官方训练场景。',
            '固定微批次2、累积2次、4轮、2,048次优化器更新。官方4,819条验证不参与晋级选择。','',
            '| 阶段 | 变体 | vehicle IoU | human IoU | 前景mIoU | 相对同阶段baseline变化 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
        for phase in ('screen','official_screen','full'):
            baseline=results[phase].get('baseline')
            for name,row in results[phase].items():
                v=row['validation']; delta=100*(v['foreground_miou']-baseline['validation']['foreground_miou']) if baseline else None
                lines.append(f"| {phase} | {name} | {100*v['classes']['vehicle']['iou']:.3f}% | {100*v['classes']['human']['iou']:.3f}% | {100*v['foreground_miou']:.3f}% | {delta:+.3f}个百分点 |" if delta is not None else f'| {phase} | {name} | 待baseline | | | |')
        lines+=['','screen是小预算实验，不能当作完整数据训练结论。official_screen是同一小预算checkpoint的官方验证复核。',
            'full从同一ImageNet起点重新训练，使用完整22,480/4,819清单、微批次2、累积4次、4轮、11,240次更新。',
            '中心2m覆盖不做实例匹配，不是检测召回。候选单项收益需结合精确率/误报检查；未提升的实验也保留。',
            '',f"按开发集规则晋级：{', '.join(results['promotions']) or '尚无'}。"]
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    try:
        dependency=Path(plan['dependency'])
        while True:
            state=json.loads((dependency/'status.json').read_text())
            if state['phase'] in ('complete','failed'):
                record('previous_training_ended',previous_state=state['phase']); break
            record('waiting_for_previous_training',dependency=str(dependency),previous_stage=state.get('stage'))
            time.sleep(30)
        source=root/'source'; source.mkdir()
        (source/'scripts').mkdir(); (source/'configs').mkdir(); (source/'resworld').mkdir()
        for path in (project/'scripts').glob('*.py'): shutil.copy2(path,source/'scripts'/path.name)
        for path in (project/'resworld').glob('*.py'): shutil.copy2(path,source/'resworld'/path.name)
        for path in (project/'configs').glob('*.json'): shutil.copy2(path,source/'configs'/path.name)
        hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}
        write_json(root/'source_manifest.json',dict(plan_sha256=plan_hash,sha256=hashes,
            data_sha256={key:hashlib.sha256(Path(plan[key]).read_bytes()).hexdigest() for key in (
                'selection_file','split_file','full_selection_file','official_split_file','initialization')}))
        def wait_memory(job):
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                if free>=args.min_free_memory_mib: return
                record('waiting_for_gpu_memory',job=job,free_memory_mib=free,required_memory_mib=args.min_free_memory_mib)
                time.sleep(30)
        def run_job(phase,name,flags):
            if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=plan_hash:
                raise ValueError('registered experiment plan changed')
            folder=root/phase; folder.mkdir(exist_ok=True)
            logs=root/'logs'; logs.mkdir(exist_ok=True)
            path=folder/name
            for attempt in range(1,4):
                wait_memory(f'{phase}/{name}')
                resume=attempt>1 and (path/'last.pt').exists() and '--eval-only' not in flags
                if attempt>1 and not resume: path=folder/f'{name}_retry{attempt}'
                command=[sys.executable,'-u','-m','scripts.train_perception_ablation',
                    '--suite-root',str(root),'--variant',name,'--output',str(path),
                    '--workers','0' if '--smoke' in flags else '4','--device','cuda',
                    '--cuda-limit-mib','10000']+flags+(['--resume'] if resume else [])
                command_path=logs/f'{phase}_{name}.attempt{attempt}.command.json'
                write_json(command_path,command)
                record('running',job=f'{phase}/{name}',attempt=attempt,output=str(path),command=command)
                child_log=logs/f'{phase}_{name}.attempt{attempt}.log'
                with child_log.open('x') as handle:
                    completed=subprocess.run(command,cwd=source,stdout=handle,stderr=subprocess.STDOUT)
                (logs/f'{phase}_{name}.attempt{attempt}.exit_code').write_text(str(completed.returncode)+'\n')
                if not completed.returncode:
                    record('job_finished',job=f'{phase}/{name}',output=str(path)); return path
                text=child_log.read_text()
                if 'out of memory' not in text.lower() or attempt==3:
                    raise RuntimeError(f'{phase}/{name} failed ({completed.returncode}); inspect {child_log}')
                record('retrying_memory_failure',job=f'{phase}/{name}',attempt=attempt,checkpoint_resume=resume)
                time.sleep(30)
            raise RuntimeError('unreachable retry state')
        for recipe in plan['variants']:
            name=recipe['name']; path=run_job('smoke',name,['--smoke'])
            results['smoke'][name]=json.loads((path/'result.json').read_text())
            refresh()
        for recipe in plan['variants']:
            name=recipe['name']; path=run_job('screen',name,['--phase','screen'])
            results['screen'][name]=training_result(path)
            if results['screen'][name]['updates']!=plan['screen']['updates']:
                raise ValueError('screen update count differs from registered budget')
            refresh()
        # Lock promotion choices before opening the official validation results.
        baseline=results['screen']['baseline']
        candidates=[name for name,row in results['screen'].items()
                    if name!='baseline' and qualifies(row,baseline,plan['promotion_rule'])]
        candidates.sort(key=lambda name:results['screen'][name]['validation']['foreground_miou'],reverse=True)
        results['promotions']=candidates[:plan['max_full_promotions']]
        write_json(root/'promotions.json',dict(selected=results['promotions'],selection_source='development only',
                   rule=plan['promotion_rule'],locked_before_official_validation=True))
        refresh()
        for recipe in plan['variants']:
            name=recipe['name']; checkpoint=results['screen'][name]['checkpoint']
            path=run_job('official_screen',name,['--phase','screen','--eval-only','--eval-split','official','--checkpoint',checkpoint])
            results['official_screen'][name]=json.loads((path/'result.json').read_text())
            refresh()
        if results['promotions']:
            for name in ['baseline']+results['promotions']:
                path=run_job('full',name,['--phase','full'])
                results['full'][name]=training_result(path)
                if results['full'][name]['updates']!=plan['full']['updates']:
                    raise ValueError('full-data update count differs from registered budget')
                refresh()
        record('complete',comparison=str(root/'comparison.json'),report=str(root/'RESULTS.md'),promotions=results['promotions'])
    except BaseException as error:
        record('failed',error=str(error)); raise
    finally:
        log.close()


if __name__=='__main__': main()
