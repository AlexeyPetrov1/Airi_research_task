"""Regressions for cadence preservation and competing independent forecasts."""
import sys
from pathlib import Path
import unittest
import subprocess
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from das_full_motion_prepare import sample_motion
from das_full_motion_diagnose import fit_motion


class FullMotionTests(unittest.TestCase):
    def test_existing_inference_is_waited_for_even_when_vram_looks_free(self):
        import das_full_motion_generate as generation
        other=SimpleNamespace(pid=12345,info={'pid':12345,'cmdline':
            ['python','-m','motion_experiments.run','--mode','inference']})
        with patch.object(generation.psutil,'process_iter',side_effect=[[other],[]]), \
                patch.object(generation.os,'getpid',return_value=67890), \
                patch.object(generation.subprocess,'check_output',return_value='11000\n'), \
                patch.object(generation.time,'sleep') as sleep, \
                patch.object(generation,'write_status') as receipt:
            generation.wait_for_gpu_before_import(Path('/tmp/unused'))
        sleep.assert_called_once_with(5)
        self.assertFalse(receipt.call_args.args[1]['own_model_loaded'])
        self.assertEqual(receipt.call_args.args[1]['other_inference_pids'],[12345])

    def test_interrupted_heartbeat_preserves_previous_valid_status(self):
        import das_full_motion_generate as generation
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);generation.write_status(out,{'stage':'previous'})
            with patch.object(generation.os,'fsync',side_effect=OSError('simulated interrupted write')):
                with self.assertRaises(OSError):generation.write_status(out,{'stage':'new'})
            self.assertEqual(json.loads((out/'resource_usage.json').read_text()),{'stage':'previous'})
            generation.write_status(out,{'stage':'new'})
            self.assertEqual(json.loads((out/'resource_usage.json').read_text()),{'stage':'new'})

    def test_real_future_access_fails_closed(self):
        source=str(Path(__file__).resolve().parents[1]/'scripts')
        program=('import sys; sys.path.insert(0,'+repr(source)+'); '
            'from das_full_motion_safety import forbid_real_future; forbid_real_future(); '
            "sys.audit('open','/tmp/runs/evaluation/future.npy','r',0)")
        result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Real future input is prohibited',result.stderr)

    def test_stretch_preserves_pose_at_matching_phase_and_delays_hold(self):
        times=np.arange(1,31)/15
        r=Rotation.from_rotvec(np.c_[times*.1,times*.2,times*.3]).as_matrix()
        t=np.c_[times**2,np.sin(times),times]
        variants=[sample_motion(r,t,timing) for timing in ['physical_2s','stretched_4s','stretched_6s']]
        for key in [0,1]:
            np.testing.assert_allclose(variants[0][key][:17],variants[1][key][:33:2],atol=1e-12)
            np.testing.assert_allclose(variants[0][key][:17],variants[2][key][::3],atol=1e-12)
        self.assertTrue(np.all(np.diff(variants[2][2])>0))
        self.assertTrue(np.all(variants[0][2][16:]==2))
        self.assertTrue(np.all(variants[1][2][32:]==2))

    def test_group00_is_independent_of_other_groups(self):
        p0=np.random.default_rng(4).normal(size=(24,3))*.05
        r=Rotation.from_euler('z',10,degrees=True).as_matrix()
        true=p0@r.T+np.array([.1,-.2,.03])
        pred=np.repeat(true[:,None],30,axis=1)
        pred[8:]+=np.array([1.,.5,-.2])
        rr,tt,residual,_=fit_motion(p0,pred,'group00')
        np.testing.assert_allclose(rr,np.repeat(r[None],30,axis=0),atol=1e-12)
        np.testing.assert_allclose(tt,np.tile([.1,-.2,.03],(30,1)),atol=1e-12)
        self.assertLess(residual.max(),1e-12)

    def test_robust_fit_rejects_small_outlier_subset(self):
        p0=np.random.default_rng(6).normal(size=(24,3))*.05
        target=p0+np.array([.12,-.08,.03])
        target[-3:]+=np.array([.4,-.5,.3])
        pred=np.repeat(target[:,None],30,axis=1)
        r,t,_,_=fit_motion(p0,pred,'robust24')
        ar,at,_,_=fit_motion(p0,pred,'all24')
        desired=p0[:-3]+np.array([.12,-.08,.03])
        robust=np.linalg.norm(p0[:-3]@r[0].T+t[0]-desired,axis=1).mean()
        plain=np.linalg.norm(p0[:-3]@ar[0].T+at[0]-desired,axis=1).mean()
        self.assertLess(robust,plain*.3)
        np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)


if __name__=='__main__':unittest.main()
