"""Run small, scene-disjoint modality/staging diagnostics; never a full train."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/city_phase1/causality_pilot')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = (project/args.output).resolve()
    if output.exists():
        raise ValueError('use a new output directory to preserve prior experiments')
    output.mkdir(parents=True)
    common = [sys.executable, '-m', 'scripts.train_staged_driving', '--selection-file',
              str(project/'runs/city_phase1/selection.json'), '--model-config',
              str(project/'configs/bev_attention_city60_pilot.json'), '--device', args.device]
    experiments = [
        ('ego', ['--stage', 'ego', '--epochs', '2']),
        ('perception', ['--stage', 'perception', '--epochs', '2']),
        ('visual_no_ego', ['--stage', 'visual', '--epochs', '2']),
        ('strict_visual_no_ego', ['--stage', 'visual', '--epochs', '2', '--strict-single-frame']),
        ('fusion', ['--stage', 'fusion', '--epochs', '1', '--perception-checkpoint',
                    str(output/'perception/best.pt'), '--ego-checkpoint', str(output/'ego/best.pt')]),
        ('joint', ['--stage', 'joint', '--epochs', '1', '--init-checkpoint', str(output/'fusion/best.pt')]),
        ('joint_scratch', ['--stage', 'joint', '--epochs', '2', '--from-scratch', '--encoder-lr-scale', '1']),
    ]
    for name, flags in experiments:
        command = common+flags+['--output', str(output/name)]
        print(json.dumps({'experiment':name, 'command':command}), flush=True)
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'finished':name}), flush=True)
    comparison = {}
    for name, _ in experiments:
        rows = [json.loads(line) for line in (output/name/'metrics.jsonl').read_text().splitlines()]
        epochs = [r for r in rows if 'validation' in r]
        best = (max(epochs, key=lambda r: r['validation']['perception']['foreground_miou']) if name == 'perception'
                else min(epochs, key=lambda r: r['validation']['final']['ade']))
        comparison[name] = dict(best_epoch=best['epoch'], updates=epochs[-1]['step'], validation=best['validation'])
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2))
    print(json.dumps({'comparison':str(output/'comparison.json')}), flush=True)


if __name__ == '__main__':
    main()
