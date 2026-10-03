"""Native observed-image MolmoPoint grounding with wrist camera provenance."""
import fmb_v2_ground as adapted
import berkeley_ground_molmopoint as original
from berkeley_preprocess import read_json

def prepare_scene(scene,processor,mode):
    inputs,pointing,info=adapted.prepare_scene(scene,processor,mode)
    info['source_camera']=read_json(scene/'metadata.json')['camera']
    original.write_json(scene/'observed/molmopoint_preparation.json',info)
    return inputs,pointing,info

if __name__=='__main__':
    original.prepare_scene=prepare_scene;original.main()
