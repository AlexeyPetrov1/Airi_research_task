"""Run the reviewed paired DaS controls sequentially on the local GPU."""
from pathlib import Path
import json, subprocess, sys
from das_prepare_control import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_molmomotion/cup/das_robot_cup'
PY='/mnt/f/AIRI_task/.venv-das/bin/python'

def main():
    for name in ['robot_coupled','robot_static']:
        out=OUT/name
        validation=json.loads((out/'control_validation.json').read_text())
        assert validation['control_sha256']==sha256(out/'control_molmomotion_720x480.mp4')
        if not validation.get('generation_ready') or validation.get('visual_review_required'):
            raise RuntimeError('Inspect prepared controls, record an actual visual_review in control_validation.json, and set generation_ready=true before running.')
        command=[PY,str(ROOT/'scripts/das_generate.py'),'--image',str(out/'image_t0_720x480.png'),
            '--tracking_path',str(out/'control_molmomotion_720x480.mp4'),
            '--prompt','Pick up the blue cup and put it into the brown cup.',
            '--checkpoint_path','/mnt/f/AIRI_task/models/Wan2.1-Fun-V1.1-1.3B-Control',
            '--output_path',str(out/'generated_molmomotion_seed42.mp4'),
            '--das_root','/mnt/f/AIRI_task/third_party/DiffusionAsShader-Wanfun',
            '--seed','42','--num_inference_steps','25','--reference-mode','clean-background',
            '--reference-image',str(OUT/'observed/clean_reference_720x480.png'),
            '--vacancy-strength','1','--vacancy-context-px','24',
            '--background-bundle',str(out/'background_completion.npz'),
            '--reference-alpha',str(OUT/'observed/clean_reference_alpha.png')]
        write_json(out/'run_command.json',{'command':command})
        code=subprocess.call([PY,str(ROOT/'scripts/das_supervise.py'),'--receipt',str(out/'process_exit.json'),'--',*command])
        if code:sys.exit(code)
        assert json.loads((out/'resource_usage.json').read_text())['success']
    write_json(OUT/'paired_generation_receipt.json',{'success':True,'order':['robot_coupled','robot_static'],'one_gpu_job_at_a_time':True})

if __name__=='__main__':main()
