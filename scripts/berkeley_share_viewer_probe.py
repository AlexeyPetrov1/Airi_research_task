"""Bounded Dataset Viewer checks for images embedded in a ShareRobot mirror."""
from pathlib import Path
import json
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'
SERVER='https://datasets-server.huggingface.co'


def get(endpoint,params):
    try:
        r=requests.get(SERVER+endpoint,params=params,timeout=(15,20))
        return r.status_code,r.json()
    except (requests.RequestException,ValueError) as e:
        return None,dict(error=repr(e),endpoint=endpoint,params=params)


def main():
    results=[]
    for dataset in ('FedorX8/ShareRobot-test','advaitgupta/sharerobot-bench','IffYuan/sharerobot_trajectory'):
        status,data=get('/splits',{'dataset':dataset})
        rec=dict(dataset=dataset,splits_status=status,splits_response=data,checks=[])
        results.append(rec)
        for split in data.get('splits',[]):
            params={k:split[k] for k in ('dataset','config','split')}
            preview_status,data=get('/first-rows',params)
            check=dict(**params,preview_status=preview_status,
                features=data.get('features'),num_rows_total=data.get('num_rows_total'),
                first_row=data.get('rows',[{}])[0].get('row') if data.get('rows') else None,
                search_results=[])
            for query in ('berkeley_autolab_ur5#episode_9/','berkeley_autolab_ur5#episode_10/'):
                status,response=get('/search',dict(**params,query=query,offset=0,length=10))
                check['search_results'].append(dict(query=query,status=status,response=response))
            rec['checks'].append(check)
            (OUT/'sharerobot_viewer_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps(dict(dataset=dataset,config=split['config'],preview_status=preview_status,
                features=[x.get('name') for x in data.get('features',[])],
                search_statuses=[x['status'] for x in check['search_results']],
                found=[x['response'].get('num_rows_total') for x in check['search_results']])),flush=True)
    (OUT/'sharerobot_viewer_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
