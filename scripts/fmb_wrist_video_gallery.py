"""Create a local reviewable HTML artifact and real-future camera clips."""
import html,json
import imageio.v2 as imageio
import numpy as np
from fmb_wrist_prepare import RUN,write,sha

def run():
    cards=[]
    for camera in ['wrist_2','wrist_1','side_1','side_2']:
        scene=RUN/camera;real=scene/'viz/real_future.mp4'
        imageio.mimwrite(real,np.load(scene/'evaluation/future_rgb.npy'),fps=10,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18'])
        cards.append(f'<article><h2>{camera}: запись FMB, t0 и2s future</h2><video controls preload="metadata" src="{camera}/viz/real_future.mp4"></video></article>')
        if camera.startswith('wrist'):
            cards.append(f'<article><h2>{camera}: MolmoMotion и reference</h2><video controls preload="metadata" src="{camera}/viz/primary_prediction_vs_real.mp4"></video><p><a href="{camera}/viz/geometry_forecasts_side_by_side.mp4">Открыть сравнение вариантов геометрии</a></p></article>')
    document='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FMB wrist — новые демонстрации</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#10151c;color:#e2e9f1}a{color:#89bdff}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}article{padding:18px;background:#1c2531;border-radius:12px}h2{font-size:18px}video{width:100%;max-height:440px;background:#000}p{max-width:900px}</style><h1>FMB wrist: прогнозы точек и реальные видео</h1><p>Source-matched episode5201. Фиолетовые точки — MolmoMotion, зелёные — будущий AllTracker reference. Прогнозы проецируются с учётом движения wrist-камеры. Официальные K служат prior; абсолютная hand-eye геометрия остаётся неопределённой. Это исследовательская демонстрация прогноза точек.</p><p><a href="../../report/fmb_wrist_v3.md">Отчёт, метрики и встроенная визуальная оценка</a> · <a href="completion_audit.json">Аудит</a></p><section class="grid">'''+''.join(cards)+'</section></html>'
    (RUN/'index.html').write_text(document,encoding='utf-8');write(RUN/'video_gallery_receipt.json',{'index_sha256':sha(RUN/'index.html'),'real_future_includes_t0':True,'source_indices':list(range(126,147)),'fps':10})
    print('Local video gallery ready',RUN/'index.html',flush=True)

if __name__=='__main__':run()
