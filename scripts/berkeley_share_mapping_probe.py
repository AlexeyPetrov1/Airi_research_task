"""Evidence for ShareRobot sequence mapping; bounded image/archive access probe."""
from pathlib import Path
from urllib.parse import quote
import io
import json
import tarfile
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'


def main():
    rev=json.loads((OUT/'sharerobot_metadata.json').read_text())['sha']
    planning=json.loads((OUT/'share_planning_task.json').read_text())
    rows=[item for item in planning['berkeley_rows'] if item['row']['id'].split('#')[-1] in ('episode_9','episode_10')]
    attempts=[]
    for ep in (9,10):
        path=f'rt_frames_success/rtx_frames_success_20/43_berkeley_autolab_ur5#episode_{ep}/frame_0.png'
        url=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/planning/images/'+quote(path,safe='/')
        r=requests.get(url,timeout=(20,60))
        attempts.append(dict(url=url,status=r.status_code,bytes=len(r.content)))
        if r.ok:(OUT/f'share_episode_{ep}_frame_0.png').write_bytes(r.content)
    part='planning/images/rt_frames_success.tar.gz.part.aa'
    url=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/{part}'
    budget=8*1024*1024
    r=requests.get(url,headers={'Range':f'bytes=0-{budget-1}'},stream=True,timeout=(20,90))
    r.raise_for_status()
    head=r.raw.read(budget);r.close()
    entries=[];probe_error=None
    try:
        with tarfile.open(fileobj=io.BytesIO(head),mode='r|gz') as archive:
            for member in archive:
                entries.append(dict(path=member.name,size=member.size))
                if len(entries)>=80:break
    except Exception as e:probe_error=repr(e)
    result=dict(revision=rev,target_episode_rows=rows,
        metadata_mapping_confirmed=bool(rows),image_mapping_confirmed=False,
        direct_image_attempts=attempts,archive_probe=dict(source=url,bytes_fetched=len(head),
        status=r.status_code,content_range=r.headers.get('Content-Range'),first_entries=entries,error=probe_error),
        limitation='Matching ShareRobot IDs/instructions do not prove RGB or temporal mapping. Pixel correspondence remains to be established.',
        required_next_evidence='Retrieve ShareRobot frames for these exact sequences and match to native/LeRobot episode frames by pixels.')
    (OUT/'sharerobot_mapping_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(rows=len(rows),image_mapping_confirmed=False,first_paths=[x['path'] for x in entries[:8]])),flush=True)


if __name__=='__main__':main()
