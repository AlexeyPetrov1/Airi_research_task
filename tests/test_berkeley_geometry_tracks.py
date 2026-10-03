import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from berkeley_geometry_infer import parse_tracks


def tracks(horizon=32,start=1):
    frames=[]
    for t in range(start,start+horizon):
        frames.append(str(t)+' '+ ' '.join(f'{p} {p*10} {-t} 2' for p in range(1,9)))
    return '<tracks coords="'+';'.join(frames)+'">8</tracks>'


class H1CompletenessTests(unittest.TestCase):
    def test_all_h1_time_and_point_ids(self):
        xyz=parse_tracks(tracks(),1,32)
        self.assertEqual(xyz.shape,(8,32,3))
        np.testing.assert_allclose(xyz[-1,-1],[.08,-.032,.002])

    def test_h3_time_ids_are_not_accepted_as_h1(self):
        with self.assertRaises(ValueError):parse_tracks(tracks(start=3),1,32)

    def test_partial_horizon_is_rejected(self):
        with self.assertRaises(ValueError):parse_tracks(tracks(horizon=31),1,32)

    def test_duplicate_point_is_rejected(self):
        bad=tracks().replace('8 80','7 80',1)
        with self.assertRaises(ValueError):parse_tracks(bad,1,32)


if __name__=='__main__':unittest.main()
