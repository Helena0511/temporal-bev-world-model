"""Independent small examples for diagnostic metric interpretation."""
import unittest
import torch
from scripts.audit_resnet_bev_quality import confusion, dilate_disk, threshold_summary
from scripts.audit_bev_perception import summarize_confusion


class BEVMetricAuditTests(unittest.TestCase):
    def test_three_classes_and_ignored_cell(self):
        target=torch.tensor([[[0,1],[2,255]]])
        prediction=torch.tensor([[[0,2],[2,1]]])
        counts=confusion(target,prediction,target!=255)
        self.assertEqual(counts.tolist(),[[1,0,0],[0,0,1],[0,0,1]])
        self.assertEqual(summarize_confusion(counts)['foreground_miou'],.25)

    def test_one_cell_shift_has_zero_iou_but_one_meter_coverage(self):
        target=torch.zeros(1,5,5,dtype=torch.bool); target[0,2,2]=True
        prediction=torch.zeros_like(target); prediction[0,2,3]=True
        self.assertEqual(int((target&prediction).sum()),0)
        self.assertTrue(bool((target&dilate_disk(prediction,1)).any()))
        self.assertTrue(bool((prediction&dilate_disk(target,1)).any()))

    def test_euclidean_tolerance_excludes_diagonal_and_does_not_wrap(self):
        mask=torch.zeros(1,5,5,dtype=torch.bool); mask[0,0,0]=True
        near=dilate_disk(mask,1)
        self.assertEqual(int(near.sum()),3)
        self.assertFalse(bool(near[0,1,1]))
        self.assertFalse(bool(near[0,-1,0]))
        self.assertTrue(bool(dilate_disk(mask,2)[0,1,1]))

    def test_probability_ranking_and_threshold_can_recover_a_perfect_binary_class(self):
        truth=torch.tensor([[[1,0],[1,2]]])
        probability=torch.zeros(1,3,2,2)
        probability[:,1]=torch.tensor([[.9,.2],[.8,.1]])
        result=threshold_summary(probability,truth,torch.ones_like(truth,dtype=torch.bool),1)
        self.assertEqual(result['positive_cells'],2)
        self.assertEqual(result['oracle_best_iou']['iou'],1)
        self.assertEqual(result['grid_binary_ap'],1)

    def test_tiny_probabilities_and_ties_are_not_lost_to_histogram_bins(self):
        truth=torch.tensor([[[1,0],[1,0]]])
        probability=torch.zeros(1,3,2,2)
        probability[:,1]=torch.tensor([[1e-6,1e-7],[1e-6,1e-7]])
        result=threshold_summary(probability,truth,torch.ones_like(truth,dtype=torch.bool),1)
        self.assertEqual(result['oracle_best_iou']['iou'],1)
        self.assertEqual(result['grid_binary_ap'],1)
        self.assertLess(result['oracle_best_iou']['threshold'],.001)


if __name__=='__main__':
    unittest.main()
