"""Contracts for paired experiments, focal loss, centers and accumulation."""
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from resworld.perception_ablation import (PerceptionCandidate, candidate_loss, focal_dice_loss,
                                initialize_candidate, make_center_targets)
from resworld.perception_objectives import bev_segmentation_loss
from tests import test_staged_driving as fixtures
from scripts.train_perception_ablation import accumulation_samples, make_loader, evaluate_candidate
from scripts.train_staged_driving import evaluate_stage
from scripts.run_perception_ablation_suite import qualifies
from scripts import train_perception_ablation as trainer


class PerceptionAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.previous)

    def fixture(self,layers=1,center=False):
        helper=fixtures.StagedDrivingTests()
        cfg=helper.config(perception='bev_query',spatial_layers=layers,use_ego_state=False)
        return PerceptionCandidate(cfg,center),helper.batch()

    def test_baseline_exactly_preserves_historical_perception_forward(self):
        model,batch=self.fixture(); model.eval()
        for key,value in model.core(batch).items():
            torch.testing.assert_close(model(batch)[key],value,rtol=0,atol=0)

    def test_new_evaluator_preserves_historical_confusion_and_depth_metric(self):
        model,batch=self.fixture(); model.eval()
        batch['bev_target'][0,2:4,2:4]=1
        batch['bev_target'][0,5,5]=2
        batch['bev_target'][0,0,0]=255
        batch['objects']=[[]]
        previous=evaluate_stage(model.core,[batch],'cpu')['perception']
        requested=evaluate_candidate(model,[batch],'cpu')
        self.assertEqual(requested['confusion'],previous['confusion'])
        self.assertEqual(requested['foreground_miou'],previous['foreground_miou'])
        self.assertEqual(requested['depth_quantized_mae_m'],previous['depth_quantized_mae_m'])

    def test_extra_layers_receive_image_conditioned_queries_and_gradients(self):
        model,batch=self.fixture(3); model.eval()
        captured=[]
        hook=model.core.visual.camera_bev.spatial[1].offsets.register_forward_pre_hook(
            lambda module,inputs:captured.append(inputs[0].detach().clone()))
        first=model(batch)
        split=len(captured)
        model({**batch,'images':torch.zeros_like(batch['images'])})
        hook.remove()
        self.assertFalse(torch.equal(captured[0],captured[split]))
        candidate_loss(first,batch)['total'].backward()
        for layer in model.core.visual.camera_bev.spatial:
            self.assertGreater(float(layer.output.weight.grad.abs().sum()),0)

    def test_controlled_initialization_copies_shared_tensors_and_records_new_layers(self):
        source,_=self.fixture(); target,_=self.fixture(3)
        initial=dict(model=source.core.state_dict(),model_config=asdict(source.config))
        result=initialize_candidate(target,initial,['spatial_layers'])
        self.assertTrue(result['new_or_shape_changed_tensors'])
        self.assertTrue(all(name.startswith(('visual.camera_bev.spatial.1.',
            'visual.camera_bev.spatial.2.')) for name in result['new_or_shape_changed_tensors']))
        torch.testing.assert_close(target.core.perception_head[0].weight,
                                   source.core.perception_head[0].weight,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'config changes'):
            initialize_candidate(target,initial,[])

    def test_zero_gamma_recovers_ce_dice_and_unknown_cells_have_no_gradient(self):
        logits=torch.randn(1,3,2,2,requires_grad=True)
        target=torch.tensor([[[0,1],[2,255]]])
        torch.testing.assert_close(focal_dice_loss(logits,target,0),
                                   bev_segmentation_loss(logits,target,'ce_dice'))
        changed=logits.detach().clone(); changed[:,:,1,1]=100
        torch.testing.assert_close(focal_dice_loss(logits,target),focal_dice_loss(changed,target))
        focal_dice_loss(logits,target).backward()
        self.assertEqual(float(logits.grad[:,:,1,1].abs().sum()),0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_center_targets_encode_metric_subcell_offset_and_collision_policy(self):
        model,_=self.fixture(); grid=model.config.grid()
        rows=[dict(xy=[1.25,-2.25],label=2)]
        target=make_center_targets(rows,grid,torch.ones(9,9,dtype=torch.bool))
        self.assertEqual(float(target['center_heatmap'][1,3,6]),1)
        torch.testing.assert_close(target['center_target_offsets'][1,:,3,6],torch.tensor([-.25,.25]))
        rows=[dict(xy=[1.1,0],label=2),dict(xy=[1.4,0],label=2)]
        target=make_center_targets(rows,grid,torch.ones(9,9,dtype=torch.bool))
        self.assertEqual(int(target['center_collisions']),1)
        self.assertEqual(int(target['center_mask'].sum()),1)
        self.assertAlmostEqual(float(target['center_target_offsets'][1,0,3,4]),-.1,places=6)

    def test_center_auxiliary_trains_shared_features_without_reading_targets(self):
        model,batch=self.fixture(center=True); model.eval()
        target=make_center_targets([dict(xy=[1.,0],label=2)],model.config.grid(),batch['bev_target'][0]!=255)
        batch.update({k:v[None] for k,v in target.items()})
        first=model(batch)
        altered={**batch,'bev_target':torch.full_like(batch['bev_target'],255),
                 'center_heatmap':torch.ones_like(batch['center_heatmap'])}
        torch.testing.assert_close(first['bev_logits'],model(altered)['bev_logits'],rtol=0,atol=0)
        candidate_loss(first,batch)['total'].backward()
        self.assertGreater(float(model.center_head[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.core.visual.camera_bev.spatial[0].output.weight.grad.abs().sum()),0)

    def test_accumulation_normalizes_short_final_window(self):
        weight=torch.tensor(1.,requires_grad=True); observed=[]
        for batch_index in range(5):
            values=torch.arange(batch_index*2+1,batch_index*2+3,dtype=torch.float32)
            loss=(weight*values).square().mean()
            (loss*2/accumulation_samples(batch_index,10,2,4)).backward()
            if (batch_index+1)%4==0 or batch_index==4:
                observed.append(float(weight.grad)); weight.grad=None
        self.assertEqual(observed,[51.,181.])

    def test_resume_data_order_is_an_exact_suffix_independent_of_model_rng(self):
        dataset=[dict(x=torch.tensor(i),objects=[]) for i in range(10)]
        first=[int(x) for batch in make_loader(dataset,2,0,123,shuffle=True) for x in batch['x']]
        torch.randn(100)
        resumed=[int(x) for batch in make_loader(dataset,2,0,123,start_batch=2,shuffle=True) for x in batch['x']]
        self.assertEqual(resumed,first[4:])

    def test_promotion_requires_both_quality_gain_and_preserved_precision(self):
        rule=dict(min_foreground_miou_gain=0.,min_human_iou_gain=0.,
                  max_vehicle_iou_drop=.01,max_human_precision_drop=.02)
        def result(vehicle,human,precision):
            return dict(validation=dict(foreground_miou=(vehicle+human)/2,
                classes=dict(vehicle=dict(iou=vehicle),human=dict(iou=human,precision=precision))))
        baseline=result(.3,.1,.25)
        self.assertTrue(qualifies(result(.3,.12,.24),baseline,rule))
        self.assertFalse(qualifies(result(.35,.1,.25),baseline,rule))
        self.assertFalse(qualifies(result(.3,.12,.1),baseline,rule))
        self.assertFalse(qualifies(result(.28,.2,.25),baseline,rule))

    def test_interrupted_training_resumes_to_identical_model_and_optimizer_steps(self):
        source,batch=self.fixture()
        config=json.loads(json.dumps(asdict(source.config)))
        class TinyBase:
            def __init__(self,root,version,split,*args,**kwargs):
                self.entries=[f'{split}{i}' for i in range(8 if split=='train' else 2)]
            def __len__(self): return len(self.entries)
            def __getitem__(self,index):
                value={k:v[0].clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
                value.update(token=self.entries[index],scene='synthetic')
                return value
            def reference_pose(self,token):
                return dict(rotation=[1.,0.,0.,0.],translation=[0.,0.,0.])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            torch.save(dict(model=source.core.state_dict(),model_config=config),root/'initialization.pt')
            (root/'selection.json').write_text(json.dumps(dict(
                train=[f'train{i}' for i in range(8)],val=[f'val{i}' for i in range(2)])))
            (root/'splits.json').write_text('{}')
            recipe=dict(name='baseline',model_config=config,image_size=[16,16],center_aux=False,
                        allowed_config_differences=[],focal=False,center_weight=.1)
            plan=dict(variants=[recipe],seed=0,data_seed=123,
                initialization=str(root/'initialization.pt'),selection_file=str(root/'selection.json'),
                split_file=str(root/'splits.json'),dataset_root='synthetic',version='v1.0-mini',
                screen=dict(epochs=2,micro_batch=2,accumulation_steps=2,effective_batch=4,updates=4))
            (root/'plan.json').write_text(json.dumps(plan))
            labels={token:[dict(rotation=[1.,0.,0.,0.],translation=[1.,0.,0.],
                size=[2.,4.,1.],label=1,visibility=4,sensor_points=5)]
                for token in [*[f'train{i}' for i in range(8)],'val0','val1']}
            def run(output,resume=False):
                argv=['train_perception_ablation.py','--suite-root',str(root),'--variant','baseline',
                      '--output',str(output),'--device','cpu','--workers','0','--threads','1']
                if resume: argv.append('--resume')
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
            real_evaluate=trainer.evaluate_candidate
            calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise RuntimeError('simulated interruption')
                return real_evaluate(*args,**kwargs)
            with patch.object(trainer,'DrivingDataset',TinyBase),patch.object(trainer,'load_labels',return_value=labels):
                run(root/'continuous')
                with patch.object(trainer,'evaluate_candidate',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        run(root/'interrupted')
                run(root/'interrupted',resume=True)
            expected=torch.load(root/'continuous/last.pt',map_location='cpu',weights_only=False)
            actual=torch.load(root/'interrupted/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(expected['step'],4); self.assertEqual(actual['step'],4)
            for name,value in expected['model'].items():
                torch.testing.assert_close(actual['model'][name],value,rtol=0,atol=0)
            for key,value in expected['optimizer']['state'].items():
                for name,tensor in value.items():
                    torch.testing.assert_close(actual['optimizer']['state'][key][name],tensor,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
