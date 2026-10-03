"""Run independent variant evaluations sequentially on a shared GPU."""
import argparse,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--tracker-repo',type=Path,required=True);p.add_argument('--tracker-checkpoint',type=Path,required=True)
p.add_argument('--variants',nargs='+',type=Path,required=True);a=p.parse_args()
scripts=Path(__file__).resolve().parent
for variant in a.variants:
    if not (variant/'generated_molmomotion_seed42.mp4').exists():
        raise FileNotFoundError(f'No completed generation in {variant}')
    commands=[['das_evaluate.py','--scene',a.scene,'--artifact-dir',variant,
               '--tracker-repo',a.tracker_repo,'--tracker-checkpoint',a.tracker_checkpoint],
              ['das_audit_duplicates.py','--scene',a.scene,'--variant',variant],
              ['das_local_quality.py','--scene',a.scene,'--variant',variant],
              ['das_verify_variant.py','--scene',a.scene,'--variant',variant]]
    for name,*arguments in commands:
        subprocess.run([sys.executable,str(scripts/name),*map(str,arguments)],check=True)
