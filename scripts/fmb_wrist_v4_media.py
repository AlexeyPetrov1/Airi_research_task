"""Render comparable original/new forecasts and a local video gallery."""
import argparse,json
import cv2,imageio.v2 as imageio,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import write,sha
from fmb_wrist_v4_diagnose import BASE,OUT

def run(provisional=False):
    prefix=('provisional_policy_' if (OUT/'motion_policy_amendment_v2.json').exists() else 'provisional_robot_') if provisional else ''
    cards=[]
    for camera in ['wrist_2','wrist_1']:
        dest=OUT/camera;viz=dest/'viz';viz.mkdir(exist_ok=True)
        metrics=json.loads((dest/(prefix+'metrics.json')).read_text());data=np.load(dest/(prefix+'predictions_and_projections.npz'))
        ref=np.load(BASE/camera/'evaluation/future_reference.npz');rgb=np.load(BASE/camera/'evaluation/future_rgb.npy')
        policy='Observed_selected_motion_policy' if 'Observed_selected_motion_policy' in metrics['methods'] else 'Robot_attached_observed_selected'
        hybrid='Selected_policy_plus_bounded_Molmo' if 'Selected_policy_plus_bounded_Molmo' in metrics['methods'] else 'Robot_plus_bounded_Molmo'
        methods=['v3_Molmo_H3',policy]
        if not provisional:methods+=[hybrid,'H3_rigid_observed','H1_native']
        titles={'v3_Molmo_H3':'Original H3','Robot_attached_observed_selected':'Robot pose forecast','Robot_plus_bounded_Molmo':'Robot pose + Molmo','Observed_selected_motion_policy':'Selected motion forecast','Selected_policy_plus_bounded_Molmo':'Selected policy + Molmo','H3_rigid_observed':'H3 clean history','H1_native':'Native H1'}
        frames=[];colors={}
        for t in range(20):
            panels=[]
            for name in [None]+methods:
                im=cv2.resize(rgb[t+1],(512,512));outside=0
                for j in range(0,24,3):
                    real=np.rint(ref['GT_2D_est'][j,t]*2).astype(int)
                    cv2.circle(im,tuple(real),3,(30,255,30),-1)
                    if name is not None:
                        p=data[name+'_uv'][j,t]*2
                        if (p>=0).all() and (p<512).all():cv2.circle(im,tuple(np.rint(p).astype(int)),5,(255,60,230),1)
                        else:
                            outside+=1
                            direction=p-real;end=real+direction*min(1,700/max(np.linalg.norm(direction),1e-12))
                            cv2.arrowedLine(im,tuple(real),tuple(np.rint(end).astype(int)),(255,60,230),1,tipLength=.08)
                cv2.rectangle(im,(0,0),(512,52),(12,18,28),-1)
                title='Real future reference' if name is None else titles[name]
                cv2.putText(im,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(im,f't={(t+1)/10:.1f}s  off image {outside}/8',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                panels.append(im)
            if not provisional:canvas=np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:6])])
            else:canvas=np.hstack(panels)
            frames.append(canvas)
        video=viz/(prefix+'forecast_comparison.mp4')
        imageio.mimwrite(video,frames,fps=10,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18'])
        for t in [0,9,19]:imageio.imwrite(viz/(prefix+f'comparison_{t+1:02d}.png'),frames[t])
        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        for name,row in metrics['methods'].items():
            if name in ['Observed_selected_motion_policy','Selected_policy_plus_bounded_Molmo']:continue
            for ax,key,scale in [(axes[0],'3D_est_m',1000),(axes[1],'2D_px',1)]:
                ax.plot(np.arange(1,21)/10,np.array(row[key]['per_time_mean'])*scale,label=titles.get(name,name))
        axes[0].set_ylabel('Estimated3D error (mm)');axes[1].set_ylabel('Image error (px)')
        for ax in axes:ax.set_xlabel('Future time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(viz/(prefix+'errors.png'),dpi=160);plt.close(fig)
        if not provisional:
            fig,axes=plt.subplots(2,3,figsize=(16,8))
            for row,name in enumerate(['H3_rigid_observed','H1_native']):
                pred=np.load(dest/'variants'/name/'future_3d.npy');times=np.arange(1,pred.shape[1]+1)/15
                for axis in range(3):
                    ax=axes[row,axis]
                    for p in pred:ax.plot(times,p[:,axis],alpha=.5,linewidth=.7)
                    ax.set_title(name+' '+['X','Y','Z'][axis]);ax.set_xlabel('Model future time (s)');ax.set_ylabel('camera_t0 (m)');ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(viz/'full_neural_outputs.png',dpi=150);plt.close(fig)
        cards.append(f'<article><h2>{camera}</h2><video controls src="{camera}/viz/{video.name}"></video><p><a href="{camera}/viz/{prefix}comparison_20.png">Кадр через2s</a></p></article>')
        print(camera,'comparison video rendered',flush=True)
    if not provisional:
        page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FMB: улучшение прогноза</title><style>body{background:#10151c;color:#e4edf6;font:16px/1.5 system-ui;margin:30px auto;max-width:1400px;padding:0 18px}a{color:#8fc2ff}video{width:100%}article{background:#1d2836;margin:22px 0;padding:18px;border-radius:12px}</style><h1>FMB: сравнение исходного и новых прогнозов</h1><p>Зелёные точки — reference на реальном будущем, фиолетовые — прогноз. Для читаемости показаны8 из24 точек; метрики используют все24. Верхний ряд: reference, исходная MolmoMotion, прогноз по прошлым TCP-позам. Нижний: TCP + ограниченная поправка MolmoMotion, H3 с очищенной историей, официальный H1.</p><p><a href="../../report/fmb_wrist_v4_improvement.md">Полный отчёт</a> · <a href="../fmb_wrist_v3/index.html">Исходный прогон</a></p>'''+''.join(cards)+'</html>'
        (OUT/'index.html').write_text(page,encoding='utf-8');write(OUT/'media_receipt.json',{'gallery_sha256':sha(OUT/'index.html'),'shown_points':8,'evaluated_points':24,'video_frames':20,'fps':10})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--provisional',action='store_true');run(p.parse_args().provisional)
