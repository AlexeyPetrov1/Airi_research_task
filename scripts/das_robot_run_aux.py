"""Targeted conditioning ablation after the paired local GPU runs."""
from pathlib import Path
import json,shutil,subprocess,time
from das_prepare_control import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_molmomotion/cup/das_robot_cup'
PY='/mnt/f/AIRI_task/.venv-das/bin/python'

def main():
    while not (OUT/'paired_generation_receipt.json').exists():time.sleep(5)
    assert json.loads((OUT/'paired_generation_receipt.json').read_text())['success']
    source=OUT/'robot_coupled';out=OUT/'robot_coupled_initial_only';out.mkdir(exist_ok=True)
    for name in ['image_t0_720x480.png','control_molmomotion_640x480.mp4','control_molmomotion_720x480.mp4',
                 'checkpoint_receipt.json','control_validation.json','control_input_freeze.json',
                 'dense_predicted_motion.npz','rigid_motion.npz','rigid_fit.json','temporal_mapping.json','control_contact_sheet.png']:
        shutil.copyfile(source/name,out/name)
    review=json.loads((out/'control_validation.json').read_text());review['variant']='robot_coupled_initial_only'
    review['conditioning_ablation']='Remove persistent empty reference and vacancy prior; keep original start+CLIP and identical coupled control'
    write_json(out/'control_validation.json',review)
    command=[PY,str(ROOT/'scripts/das_generate.py'),'--image',str(out/'image_t0_720x480.png'),
        '--tracking_path',str(out/'control_molmomotion_720x480.mp4'),
        '--prompt','Pick up the blue cup and put it into the brown cup.',
        '--checkpoint_path','/mnt/f/AIRI_task/models/Wan2.1-Fun-V1.1-1.3B-Control',
        '--output_path',str(out/'generated_molmomotion_seed42.mp4'),
        '--das_root','/mnt/f/AIRI_task/third_party/DiffusionAsShader-Wanfun',
        '--seed','42','--num_inference_steps','25','--reference-mode','initial-only']
    write_json(out/'run_command.json',{'command':command,'reason':'B full vacancy prior produced transparency and loss of cup identity; test identical motion without persistent empty reference or latent vacancy prior'})
    code=subprocess.call([PY,str(ROOT/'scripts/das_supervise.py'),'--receipt',str(out/'process_exit.json'),'--',*command])
    raise SystemExit(code)

if __name__=='__main__':main()
