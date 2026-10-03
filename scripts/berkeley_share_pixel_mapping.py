"""Prove ShareRobot target sequence identity against the original RLDS RGB.

Reuses the already verified TFRecord shard; no downloads or model inference.
Exact decoded-pixel SHA256 matching is decisive. Approximate frame searches
are recorded separately and cannot pass the exact proof criterion.
"""
from pathlib import Path
import hashlib,io,json,struct
import numpy as np
from PIL import Image,ImageDraw
from probe_fmb_rlds_record import find_feature_values

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/berkeley_ur5_molmomotion'
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'
DATA=ROOT.parent/'data/berkeley_ur5_two_scenes'


def digest(pixels):return hashlib.sha256(pixels.tobytes()).hexdigest()


def main():
    original_receipt=json.loads((BASE/'sources/native_observed_receipt.json').read_text())
    shard=DATA/'berkeley_autolab_ur5-train.tfrecord-00004-of-00412'
    raw=shard.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=original_receipt['shard_sha256']:
        raise ValueError('Native source shard bytes changed')
    records=[];pos=0
    while pos<len(raw):
        size=struct.unpack('<Q',raw[pos:pos+8])[0];pos+=12
        records.append(raw[pos:pos+size]);pos+=size+4
    del raw
    results=[]
    for name,ep,record_idx in (('bottle',9,0),('cup',10,1)):
        paths=sorted((OUT/'share_exact_frames'/f'episode_{ep}').glob('frame_*.png'))
        if len(paths)!=30:raise ValueError(f'Need all 30 authentic PNGs: {name}')
        record=records[record_idx];features=dict(find_feature_values(record))
        native=[];hashes={}
        for idx,(start,end) in enumerate(features['steps/observation/image']):
            pixels=np.asarray(Image.open(io.BytesIO(record[start:end])).convert('RGB'))
            native.append(pixels);hashes.setdefault(digest(pixels),[]).append(idx)
        small=np.stack([np.asarray(Image.fromarray(x).resize((64,48))) for x in native]).astype(float)
        matches=[];source_ids=[];t0=original_receipt['scenes'][name]['t0_source_frame']
        for path in paths:
            pixels=np.asarray(Image.open(path).convert('RGB'));h=digest(pixels);exact=hashes.get(h,[])
            frame=int(path.stem.split('_')[1])
            item=dict(share_frame=frame,png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decoded_pixel_sha256=h,exact_native_source_indices=exact)
            if len(exact)==1:
                source=exact[0];source_ids.append(source)
                item.update(native_frame=source,relative_to_t0_s=(source-t0)/5,
                    pixel_max_abs_difference=0,pixel_rmse=0.)
            else:
                sample=np.asarray(Image.fromarray(pixels).resize((64,48))).astype(float)
                distances=np.mean((small-sample)**2,axis=(1,2,3));best=int(np.argmin(distances))
                item.update(approximate_nearest_native_frame=best,
                    nearest_full_rgb_rmse=float(np.sqrt(np.mean((pixels.astype(float)-native[best])**2))),
                    exact_unique_pixel_match=False)
            matches.append(item)
        exact_all=len(source_ids)==30
        monotonic=exact_all and bool(np.all(np.diff(source_ids)>0))
        result=dict(scene=name,source_episode=ep,
            share_sequence=f'rtx_frames_success_20/43_berkeley_autolab_ur5#episode_{ep}',
            native_record_in_shard=record_idx,native_episode_frames=len(native),t0_source_frame=t0,
            all_30_frames_exact_unique_pixel_match=exact_all,temporal_order_monotonic=monotonic,
            matched_source_indices=source_ids,matches=matches,
            baseline_native_identity_receipt='../../berkeley_ur5_molmomotion/sources/native_observed_receipt.json')
        if exact_all:
            result['uniform_floor_linspace_indices_match']=bool(np.array_equal(source_ids,np.linspace(0,len(native)-1,30,dtype=int)))
            result['uniform_rounded_linspace_indices_match']=bool(np.array_equal(source_ids,np.linspace(0,len(native)-1,30).round().astype(int)))
            result['rounded_linspace_31_drop_last_indices_match']=bool(np.array_equal(
                source_ids,np.linspace(0,len(native)-1,31).round().astype(int)[:-1]))
            result['sampling_rule_inference']='Observed indices match round(linspace(0,N-1,31))[:-1]. This is a reconstruction from exact matches, not a claim to have retrieved the author extraction implementation.'
            # Several positions around the actual experiment window, paired
            # with independent native source pixels. No image editing/overlays.
            chosen=[int(np.argmin(np.abs(np.array(source_ids)-target))) for target in (t0-2,t0,t0+5,t0+10)]
            canvas=Image.new('RGB',(4*320,2*264),'white');draw=ImageDraw.Draw(canvas)
            for col,index in enumerate(chosen):
                source=source_ids[index]
                for row,im in enumerate((Image.open(paths[index]).convert('RGB'),Image.fromarray(native[source]))):
                    x=320*col;y=264*row;canvas.paste(im.resize((320,240)),(x,y))
                    caption=f'{"ShareRobot" if row==0 else "Native RLDS"}: {index if row==0 else source}; dt={(source-t0)/5:+.1f}s'
                    draw.text((x+4,y+242),caption,fill='black')
            canvas.save(OUT/f'share_native_pixel_pairs_{name}.png')
        results.append(result)
        print(json.dumps(dict(scene=name,all_30_exact=exact_all,monotonic=monotonic,
                              source_indices=source_ids)),flush=True)
    author=json.loads((OUT/'sharerobot_archive_headers.json').read_text())['upstream_mapping_issue']['comments']
    proof=dict(image_mapping_confirmed=all(x['all_30_frames_exact_unique_pixel_match'] and x['temporal_order_monotonic'] for x in results),
        share_revision=json.loads((OUT/'sharerobot_metadata.json').read_text())['sha'],
        native_source=original_receipt['source'],native_shard_sha256=original_receipt['shard_sha256'],
        mapping_rule_primary_source='https://github.com/FlagOpen/ShareRobot/issues/4#issuecomment-3094378733',
        mapping_rule_author=author[0]['user']['login'],scenes=results,
        future_used_for_new_model_inputs=False,
        caveat='ShareRobot planning retains 30 sampled frames over the whole episode, not the dense 5 FPS model history or future. Model inputs remain the original verified native/LeRobot stream; mapping does not substitute sparse ShareRobot frames into H3.')
    (OUT/'sharerobot_pixel_mapping.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if not proof['image_mapping_confirmed']:raise ValueError('Exact pixel identity proof did not pass')


if __name__=='__main__':main()
