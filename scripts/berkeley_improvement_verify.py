"""Independent CPU audit of local evidence; Git publication is checked separately."""
from pathlib import Path
import argparse,hashlib,json,re
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1'
BASE=ROOT/'runs/berkeley_ur5_molmomotion'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final',action='store_true');args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,expected in protocol['source_tree_sha256'].items():
        if sha(BASE/path)!=expected:raise ValueError(f'Original artifact changed: {path}')
    correction=[]
    for scene in ('cup','bottle'):
        original=np.load(BASE/scene/'predictions/future_3d.npy').astype(float)
        p0=np.load(BASE/scene/'observed/points_3d_history.npy')[-1].astype(float)
        actual=np.load(OUT/scene/'displacement_one_third_full_3d.npy')
        if not np.array_equal(actual,p0[:,None]+(original-p0[:,None])/3):
            # Multiplication by 1/3 versus division differs by float64 roundoff.
            np.testing.assert_allclose(actual,p0[:,None]+(original-p0[:,None])/3,atol=1e-14,rtol=0)
        correction.append(scene)
    completed=[];pending=[]
    for receipt in OUT.glob('CASE-*/**/predictions/group_*/model_run.json'):
        status=json.loads(receipt.read_text())
        if not status.get('success'):
            pending.append(str(receipt.relative_to(OUT)));continue
        group=receipt.parent.name;scene=receipt.parents[2]
        expected_shape=(8,status.get('future_horizon',30),3)
        xyz=np.load(receipt.parent/'future_3d.npy')
        if xyz.shape!=expected_shape or not np.isfinite(xyz).all():raise ValueError('Incomplete real output')
        raw=(receipt.parent/'raw_model_output.txt').read_text()
        match=re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*',raw)
        if not match:raise ValueError('Malformed tracks text')
        frames=match.group(1).split(';');H=status.get('history_size',3)
        parsed=np.empty_like(xyz)
        if len(frames)!=expected_shape[1]:raise ValueError('Partial horizon')
        for step,frame in enumerate(frames):
            tokens=frame.split()
            if len(tokens)!=33 or float(tokens[0])!=step+H:raise ValueError('Incorrect timestamp IDs')
            record=np.array([int(x) for x in tokens[1:]]).reshape(8,4)
            if sorted(record[:,0].tolist())!=list(range(1,9)):raise ValueError('Missing/duplicate point')
            parsed[record[:,0]-1,step]=record[:,1:]/1000
        anchor=np.load(scene/f'groups/{group}/points_3d_history.npy')[-1,0]
        np.testing.assert_allclose(xyz,parsed+anchor,atol=1e-4,rtol=0)
        for filename,key in [('future_3d.npy','prediction_sha256'),('raw_model_output.txt','raw_output_sha256')]:
            if sha(receipt.parent/filename)!=status[key]:raise ValueError('Prediction bytes changed')
        completed.append(str(receipt.relative_to(OUT)))
    movies=[]
    for movie in OUT.rglob('*comparison.mp4'):
        cap=cv2.VideoCapture(str(movie));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if len(frames)!=10 or abs(fps-5)>1e-3:raise ValueError('Invalid scientific movie cadence')
        audit=movie.parent/'video_audit';audit.mkdir(exist_ok=True)
        # Inspect all ten frames of the corrected column, not just endpoints.
        selected_column=1 if movie.name=='fixed_comparison.mp4' else 2
        tiles=[]
        for i,frame in enumerate(frames):
            cv2.imwrite(str(audit/f'{movie.stem}_frame_{i:02d}.jpg'),frame)
            width=frame.shape[1]//3
            tile=cv2.resize(frame[:,selected_column*width:(selected_column+1)*width],(320,240))
            tiles.append(tile)
        sheet=np.vstack([np.hstack(tiles[:5]),np.hstack(tiles[5:])])
        cv2.imwrite(str(audit/f'{movie.stem}_all10_corrected.jpg'),sheet)
        movies.append(dict(path=str(movie.relative_to(OUT)),frames=10,fps=fps,sha256=sha(movie)))
    final_checks={}
    if args.final:
        if pending or len(completed)!=19:raise ValueError('Expected all 19 real groups, with no pending attempt')
        for case in ('CASE-AUGE','CASE-NOK','CASE-NOK-STRICT'):
            for name in ('cup','bottle'):
                scene=OUT/case/name
                status=json.loads((scene/'predictions/model_run.json').read_text())
                if not status.get('success') or status.get('mode')!='full_fixed_24':raise ValueError('Incomplete full case')
                groups=[np.load(scene/f'predictions/group_{g:02d}/future_3d.npy') for g in range(3)]
                np.testing.assert_array_equal(np.load(scene/'predictions/future_3d.npy'),np.concatenate(groups))
                ids=np.load(scene/'observed/selected_point_ids.npy')
                np.testing.assert_array_equal(ids,np.load(BASE/name/'observed/selected_point_ids.npy'))
                frozen=json.loads((scene/'predictions/input_freeze.json').read_text())
                for path,expected in frozen['sha256'].items():
                    if sha(scene/path)!=expected:raise ValueError(f'Case input changed after freeze: {case}/{name}/{path}')
                for path in ('observed/native_depth.npy','observed/observed_tracks_2d.npz','observed/mask.png'):
                    if sha(scene/path)!=sha(BASE/name/path):raise ValueError('Frozen physical inputs changed')
                provenance=json.loads((scene/'geometry/case_provenance.json').read_text())
                if provenance['future_used'] is not False:raise ValueError('Noncausal geometry')
                if case=='CASE-NOK-STRICT':
                    strict=json.loads((scene/'geometry/strict_geometry_audit.json').read_text())
                    if strict['original_K_read_for_geometry'] or strict['original_XYZ_read_for_geometry'] or strict['original_trust_weights_reused']:
                        raise ValueError('External geometry dependency remains')
        mapping=json.loads((OUT/'sources/sharerobot_pixel_mapping.json').read_text())
        if not mapping['image_mapping_confirmed']:raise ValueError('Mapping not proved')
        from berkeley_share_extract_range import check_png
        accepted=[]
        for extraction in (OUT/'sources').glob('share_extract_*.json'):
            accepted+=json.loads(extraction.read_text())['accepted_images']
        if len(accepted)!=60:raise ValueError('Incomplete image evidence')
        for item in accepted:
            p=OUT/'sources'/item['local_path']
            if sha(p)!=item['sha256']:raise ValueError('ShareRobot evidence bytes changed')
            check_png(p.read_bytes())
        review=json.loads((OUT/'visual_review_final.json').read_text())
        if review['status']!='complete' or not review['all_required_groups_reviewed']:raise ValueError('Visual review unfinished')
        final_checks=dict(full_24_ID_cases=6,case_input_freezes_verified=True,
            strict_geometry_receipts_verified=True,ShareRobot_exact_image_files_checked=60,
            visual_review_complete=True,Kalib='Applicability audited; actual calibration not performed without verified TCP correspondences')
    result=dict(completed_artifact_checks_passed=True,publication_checked=False,
        original_artifacts_unchanged=len(protocol['source_tree_sha256']),correction_checked=correction,
        completed_actual_model_groups=completed,pending_groups=pending,movies=movies,
        local_experiment_requirements_verified=args.final,final_checks=final_checks,
        required_next='Final report publication and remote Git verification' if args.final else
            'Visual review of full 24-ID cases and strict no-prior-K pilot; complete ShareRobot image mapping; final report and publication')
    (OUT/('final_verification.json' if args.final else 'progress_verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verified_groups=len(completed),pending_groups=len(pending),decoded_movies=len(movies),local_experiment_verified=args.final)),flush=True)


if __name__=='__main__':main()
