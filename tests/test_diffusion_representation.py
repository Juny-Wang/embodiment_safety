import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from algos.diffusion_policy.representation import encode_state, state_config, validate_config, ROT6D, LEGACY

class RepresentationTests(unittest.TestCase):
    def test_equivalent_signs_and_rotation_columns(self):
        q=Rotation.random(200,random_state=7).as_quat()
        a=np.random.default_rng(8).normal(size=(200,31)).astype(np.float32);a[:,22:26]=q
        b=a.copy();b[:,22:26]*=-1
        out=encode_state(a,ROT6D)
        np.testing.assert_array_equal(out,encode_state(b,ROT6D))
        matrix=Rotation.from_quat(a[:,22:26]).as_matrix()
        np.testing.assert_allclose(out[:,22:25],matrix[:,:,0],atol=1e-7)
        np.testing.assert_allclose(out[:,25:28],matrix[:,:,1],atol=1e-7)
        np.testing.assert_array_equal(out[:,:22],a[:,:22])
        np.testing.assert_array_equal(out[:,28:],a[:,26:])
    def test_pi_crossing_and_history_batch_shape(self):
        a=np.zeros((1,2,31),np.float32)
        a[0,:,22:26]=Rotation.from_rotvec([[np.pi-1e-5,0,0],[np.pi+1e-5,0,0]]).as_quat()
        a[0,1,22:26]*=-1
        out=encode_state(a,ROT6D)
        self.assertEqual(out.shape,(1,2,33))
        self.assertLess(np.linalg.norm(out[0,1]-out[0,0]),3e-5)
    def test_legacy_and_rejection(self):
        a=np.zeros(31,np.float32);a[25]=1
        np.testing.assert_array_equal(encode_state(a,LEGACY),a)
        validate_config({"state_dim":31})
        validate_config(state_config(ROT6D))
        with self.assertRaises(ValueError):validate_config({"state_dim":31,"state_representation":ROT6D})
        with self.assertRaises(ValueError):encode_state(np.zeros(31),ROT6D)
        with self.assertRaises(ValueError):encode_state(np.zeros(33),ROT6D)

if __name__=="__main__":unittest.main()
