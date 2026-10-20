import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dehaze.model import AODNet
from dehaze.metrics import scores
from dehaze.checkpoint import save, load
from dehaze.data import ROOT, build_manifest, validate_manifest, sample_batch

class DehazingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)
    def test_identity_initialization(self):
        model=AODNet();model.initialize_identity();x=torch.rand(1,3,19,23)
        torch.testing.assert_close(model(x),x,atol=1e-7,rtol=0)
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in AODNet().parameters()),1761)
    def test_identity_metrics(self):
        x=torch.rand(1,3,20,21);m=scores(x,x)
        self.assertEqual(m['psnr_db'],float('inf'));self.assertAlmostEqual(m['ssim'],1.,places=10)
    def test_known_psnr(self):
        x=torch.zeros(1,3,16,16);y=torch.ones_like(x)*.1
        self.assertAlmostEqual(scores(x,y)['psnr_db'],20,places=5)
    def test_ssim_symmetry_and_bad_size(self):
        a,b=torch.rand(1,3,18,19),torch.rand(1,3,18,19)
        self.assertAlmostEqual(scores(a,b)['ssim'],scores(b,a)['ssim'],places=10)
        with self.assertRaises(ValueError):scores(a[:,:,:5,:5],b[:,:,:5,:5])
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            m=AODNet();m.eval();p=Path(d)/'m.pt';save(p,m,{'seed':2026,'torch':torch.__version__});n,meta=load(p);torch.load(p,weights_only=True)
            x=torch.rand(1,3,17,20);torch.testing.assert_close(m(x),n(x));self.assertEqual(meta['seed'],2026)
    def test_training_step_changes_weights(self):
        m=AODNet();m.initialize_identity();optimizer=torch.optim.Adam(m.parameters(),lr=.001)
        x=torch.rand(2,3,16,16);loss=((m(x)-x*.8)**2).mean();loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float((m(x)-x).abs().max().detach()),0)
    def test_scene_split_reproducibility(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as d:
            base=Path(d)
            for kind in ('hazy','clear'):
                (base/kind).mkdir()
                for i in range(1,46):Image.new('RGB',(12,12),(i,20,30)).save(base/kind/f'{i:02d}_{kind}.png')
            a=build_manifest(base,base/'a.json');b=build_manifest(base,base/'b.json')
            self.assertEqual(a,b);self.assertEqual(sum(r['split']=='test' for r in a['rows']),5)
            a['rows'][1]['scene_id']=a['rows'][0]['scene_id']
            with self.assertRaises(ValueError):validate_manifest(a)
    def test_missing_pair_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as d:
            base=Path(d);(base/'hazy').mkdir();(base/'clear').mkdir()
            Image.new('RGB',(12,12)).save(base/'hazy/01_hazy.png')
            with self.assertRaises(ValueError):build_manifest(base,base/'manifest.json')
    def test_paired_augmentation(self):
        x=torch.rand(3,20,22);cache=[('a',x,x.clone())]
        a,b=sample_batch(cache,random.Random(7),4,12);torch.testing.assert_close(a,b)
        self.assertEqual(tuple(a.shape),(4,3,12,12))

if __name__=='__main__': unittest.main()
