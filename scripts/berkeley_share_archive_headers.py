"""Check whether ShareRobot's split archive parts are separate gzip streams."""
from pathlib import Path
import json,time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'


def main():
    meta=json.loads((OUT/'sharerobot_metadata.json').read_text());rev=meta['sha']
    names=sorted(x['rfilename'] for x in meta['siblings']
        if x['rfilename'].startswith('planning/images/rt_frames_success.tar.gz.part.'))
    def probe(name):
        url=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/'+quote(name,safe='/')
        started=time.monotonic()
        with requests.get(url,headers={'Range':'bytes=0-63'},stream=True,timeout=(20,60)) as r:
            r.raise_for_status();head=r.raw.read(64)
            return dict(path=name,status=r.status_code,bytes_read=len(head),
                content_range=r.headers.get('Content-Range'),header_hex=head.hex(),
                gzip_header=head.startswith(b'\x1f\x8b'),elapsed_s=time.monotonic()-started)
    with ThreadPoolExecutor(max_workers=4) as pool:
        probes=list(pool.map(probe,[names[0],names[1],names[20],names[-1]]))
    mirrors=requests.get('https://huggingface.co/api/datasets',
        params={'search':'ShareRobot','limit':100},timeout=30)
    mirrors.raise_for_status()
    issues=requests.get('https://api.github.com/repos/FlagOpen/ShareRobot/issues/4',timeout=30)
    comments=requests.get('https://api.github.com/repos/FlagOpen/ShareRobot/issues/4/comments',timeout=30)
    result=dict(revision=rev,archive_part_count=len(names),part_header_probes=probes,
        independently_compressed_parts=all(p['gzip_header'] for p in probes),
        dataset_mirrors=[x['id'] for x in mirrors.json()],
        upstream_mapping_issue=dict(url='https://github.com/FlagOpen/ShareRobot/issues/4',
            status=issues.status_code,state=issues.json().get('state'),
            comments_status=comments.status_code,comments=comments.json()),
        limitation='The author reply says source episode IDs are preserved and 30 frames are evenly extracted across datasets. This supports the naming rule but does not alone prove Berkeley pixel identity; see sharerobot_pixel_mapping.json for the independent exact-image proof.')
    (OUT/'sharerobot_archive_headers.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(probes=probes,mirrors=result['dataset_mirrors'],
                         mapping_issue_comments=len(comments.json()))),flush=True)


if __name__=='__main__':main()
