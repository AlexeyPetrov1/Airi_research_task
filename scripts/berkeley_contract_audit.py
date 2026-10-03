"""Source-grounded audit of ordinal model time and prepared-case provenance."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1'


def main():
    paths=['src/molmo_motion/processor.py','src/molmo_motion/data/trajectory_3d_dataset.py',
           'data/checkpoints/MolmoMotion-4B-H3-F30/config.yaml']
    sources=[]
    for name in paths:
        p=ROOT/name; text=p.read_text(encoding='utf8')
        needles=['timestamps = np.arange','timestamps_arr = np.arange','target_fps=1.0',
                 'hist_timestamps =','future_timestamps =','time_mode: per-frame-compact',
                 'pre-resampled to 15 fps','retains its 4× temporal stride']
        lines=[dict(line=i+1,text=line.strip()) for i,line in enumerate(text.splitlines())
               if any(needle in line for needle in needles)]
        sources.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence=lines))
    result=dict(sources=sources,processor_matches_training_ordinal_time=True,
        physical_seconds_argument_available=False,history_time_labels=[0.,1.,2.],
        first_future_time_label=3.,model_future_step_duration_assumed_s=1/15,
        real_Berkeley_history_step_duration_s=.2,
        finding='Public processor and training builder both use ordinal frame labels and synthetic video target_fps=1. This is not evidence of a processor-only 1 FPS bug.',
        limitation='15 FPS is the documented inference interpretation and DROID training track rate; source code also handles other datasets and temporal strides. It does not prove every training sample has one identical physical rate.',
        causal_conclusion='Cadence mismatch remains a hypothesis. Fixed 1/3 scaling, timewarp, oracle scale and H1 are separate controls; none alone establishes the cause.')
    dest=OUT/'sources/model_time_contract.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    # Only fix descriptive metadata before any strict-case input is sealed.
    corrected=[]
    for scene in ('cup','bottle'):
        case=OUT/'CASE-NOK-STRICT'/scene
        if (case/'predictions/input_freeze.json').exists():
            raise ValueError('Strict case is sealed; do not edit its metadata')
        p=case/'geometry/case_provenance.json'; record=json.loads(p.read_text())
        record['unchanged']=[x for x in record['unchanged'] if x!='smoothed Z']
        record['smoothed_Z_recomputed_with_independent_rays']=True
        p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        corrected.append(scene)
    print(json.dumps(dict(time_contract_audited=True,strict_unsealed_provenance_corrected=corrected)),flush=True)


if __name__=='__main__':main()
