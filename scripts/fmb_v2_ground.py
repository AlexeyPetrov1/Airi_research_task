"""Use the Berkeley MolmoPoint runner with a FMB observed-only prompt adapter."""
import sys
from datetime import datetime, timezone
import cv2
import numpy as np
import berkeley_ground_molmopoint as original
from berkeley_preprocess import load_observed

def prepare_scene(scene, processor, mode):
    import torch
    if (scene/'predictions/input_freeze.json').exists() or (scene/'observed/molmopoint_grounding.json').exists():
        raise RuntimeError('Refusing to overwrite grounding or frozen inputs')
    assert mode == 'image'
    rgb, indices, metadata = load_observed(scene)
    prompt = 'point to the red rectangular peg held by the robot gripper, above the blue board'
    image_path = scene/'observed/molmopoint_t0.png'
    cv2.imwrite(str(image_path), rgb[-1][...,::-1])
    messages = [{'role':'user','content':[{'type':'text','text':prompt},
                 {'type':'image','image':str(image_path.resolve())}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_tensors='pt', return_dict=True, padding=True, return_pointing_metadata=True)
    pointing = original.numpy_metadata(inputs.pop('metadata'))
    assert len(pointing['image_sizes']) == 1
    np.save(scene/'observed/molmopoint_input_ids.npy', inputs['input_ids'].cpu().numpy())
    info = {'scene':scene.name,'source_frame':int(indices[-1]),'source_camera':'side_1',
            'object_phrase':metadata['target_object'],'prompt':prompt,'future_used':False,
            'input_modality':'image','input_real_frame_count':1,'input_source_indices':[int(indices[-1])],
            'model_input_rgb_sha256':original.sha256(image_path),
            'observed_rgb_sha256':original.sha256(scene/'observed/rgb.npy'),
            'input_shapes':{k:list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},
            'created_utc':datetime.now(timezone.utc).isoformat()}
    original.write_json(scene/'observed/molmopoint_preparation.json',info)
    return inputs,pointing,info

if __name__ == '__main__':
    original.prepare_scene = prepare_scene
    original.main()
