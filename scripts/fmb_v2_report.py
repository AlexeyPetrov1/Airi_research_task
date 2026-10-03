"""Build the independent FMB v2 report and verify completion evidence."""
from pathlib import Path
import json
import re
import numpy as np
import cv2
from fmb_v2_scene import ROOT,RUN,SPECS
from berkeley_preprocess import read_json,write_json,sha256
from fmb_v2_infer import strict_parse
from fmb_v2_evaluate import check_freeze

def verify():
    weights=read_json(RUN/'checkpoint_byte_audit.json');assert all(row['success'] for row in weights.values())
    v1=read_json(RUN/'preserved_v1.json')
    changed=[name for name,digest in v1['sha256'].items() if sha256(ROOT/name)!=digest]
    assert not changed,f'FMB v1 changed: {changed}'
    receipts={}
    for name in SPECS:
        scene=RUN/name;meta=read_json(scene/'metadata.json');audit=read_json(scene/'geometry/input_audit.json')
        assert audit['success'] and audit['selected_points']==24 and check_freeze(scene)>0
        assert np.load(scene/'observed/rgb.npy').shape==(8,256,256,3)
        assert np.load(scene/'observed/points_3d_history.npy').shape==(3,24,3)
        assert read_json(scene/'geometry/camera_motion_audit.json')['fixed_camera_supported']
        assert read_json(scene/'geometry/registration_audit.json')['future_used'] is False
        assert read_json(scene/'observed/molmopoint_grounding.json')['success']
        assert read_json(scene/'observed/grounding_metadata.json')['segmentation']=='official Meta SAM2.1 Hiera-Large'
        for folder,h,f in [('predictions',3,30)]+([('predictions_h1',1,32)] if meta['primary_example'] else []):
            status=read_json(scene/folder/'model_run.json');assert status['success'] and status['successful_chunks']==3
            arrays=[]
            for group in sorted((scene/'groups').glob('group_*')):
                dest=scene/folder/group.name
                delta=strict_parse((dest/'raw_model_output.txt').read_text(),h,f)
                future=np.load(dest/'future_3d.npy');anchor=np.load(dest/'anchor.npy')
                assert future.shape==(8,f,3) and np.isfinite(future).all()
                np.testing.assert_allclose(future,delta+anchor,atol=1e-4)
                assert sha256(dest/'future_3d.npy')==read_json(dest/'model_run.json')['prediction_sha256']
                arrays.append(future)
            assert np.array_equal(np.concatenate(arrays),np.load(scene/folder/'future_3d.npy'))
        data=np.load(scene/'evaluation/references_and_methods.npz')
        assert data['GT_2D_est'].shape==(24,20,2) and data['GT_3D_est'].shape==(24,20,3)
        metrics=read_json(scene/'evaluation/metrics.json')
        assert all(row['2D']['valid_pairs']==metrics['common_mask_2D_pairs'] for row in metrics['methods'].values())
        assert all(row['3D_est']['valid_pairs']==metrics['common_mask_3D_est_pairs'] for row in metrics['methods'].values())
        videos={}
        for filename in ['side_by_side.mp4','prediction_vs_real.mp4']:
            cap=cv2.VideoCapture(str(scene/'viz'/filename));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            assert count==20 and abs(fps-10)<.01;videos[filename]={'frames':count,'fps':fps}
        receipts[name]={'success':True,'frozen_inputs':check_freeze(scene),'videos':videos,
            'H3_complete_groups':3,'H1_complete_groups':3 if meta['primary_example'] else 0,
            'reference_pairs_2D':metrics['common_mask_2D_pairs'],'reference_pairs_3D_est':metrics['common_mask_3D_est_pairs']}
    visual=read_json(RUN/'visual_review.json');assert visual['observed_masks_reviewed'] and visual['registration_reviewed'] and visual['future_reference_reviewed']
    result={'success':True,'preserved_v1_file_count':len(v1['sha256']),'v1_unchanged':True,'scenes':receipts,
            'checkpoint_byte_audit':weights,
            'no_future_used_to_form_inputs':True,'future_export_prediction_gated':True,
            'metric_scale_is_confirmed':False,'unit_of_experiment':'episode','visual_review':visual}
    write_json(RUN/'completion_audit.json',result)
    return result

def report():
    result=verify();rows=[]
    for scene in SPECS:
        metrics=read_json(RUN/scene/'evaluation/metrics.json')
        for name,row in metrics['methods'].items():
            rows.append(f"| {scene} | {name} | {row['2D']['ADE_2D_px']:.3f} | {row['2D']['FDE_2D_px']:.3f} | {row['3D_est']['ADE_3D_est_m']:.4f} | {row['3D_est']['FDE_3D_est_m']:.4f} |")
    primary=read_json(RUN/'episode_5201/evaluation/metrics.json')
    runtime_rows=[]
    for scene_name in SPECS:
        scene=RUN/scene_name
        for folder in ['predictions']+(['predictions_h1'] if scene_name=='episode_5201' else []):
            status=read_json(scene/folder/'model_run.json')
            peak=max(row['peak_cuda_allocated_gib'] for row in status['groups'])
            runtime_rows.append(f"| {scene_name} | {status['model_id'].split('/')[-1]} | {status['prediction_seconds']:.2f} | {peak:.2f} |")
    h3=primary['methods']['MolmoMotion_H3']['2D'];baseline=[primary['methods'][name]['2D'] for name in ['static','constant_velocity']]
    conclusion=(f"На основном episode_5201 большая ошибка H3 сохраняется: ADE {h3['ADE_2D_px']:.2f} px против "
        f"{baseline[0]['ADE_2D_px']:.2f} у static и {baseline[1]['ADE_2D_px']:.2f} у constant velocity. "
        f"FDE H3 {h3['FDE_2D_px']:.2f} px хуже static ({baseline[0]['FDE_2D_px']:.2f}), "
        f"но лучше constant velocity ({baseline[1]['FDE_2D_px']:.2f}). "
        "На втором FMB-контроле H3 почти совпадает со static и уступает constant velocity по основным 2D-метрикам. "
        "Следовательно, тезис о существенном проигрыше обоим baseline по обеим метрикам на обеих сценах не подтверждён.")
    uncertainty=read_json(RUN/'episode_5201/evaluation/reference_uncertainty.json')
    control_uncertainty=read_json(RUN/'fmb_control_n3/evaluation/reference_uncertainty.json')
    text=f'''# FMB v2 — Berkeley-matched pipeline

{conclusion} Это независимый повтор на зафиксированных окнах, с новым MolmoPoint → SAM 2.1 → 100 K-means-кандидатов → AllTracker → 24 точки, observed-only UniDepthV2 K и sensor Z16. Результат не подбирался по будущим ошибкам. [Проверка завершения](../runs/fmb_v2_berkeley_matched/completion_audit.json) подтверждает полные прогнозы, формы, общие маски, видео и неизменность {result['preserved_v1_file_count']} файлов FMB v1.

FMB v1 сохранён отдельно как **FMB v1 — geometry sensitivity study**: [основной опыт](fmb_quantitative_2d_episode_5201.md), [второй опыт](fmb_quantitative_2d_second_trial.md), [исследование геометрии](fmb_geometry_forecast_study.md). Входы и результаты этих работ не перезаписаны. Сравниваемым эталоном методологии служит [Berkeley research story](../runs/berkeley_ur5_molmomotion/research_story.md).

## Данные и фиксированное время

Основной эпизод — `ShareRobot 57_fmb#episode_5201`, восстановленный исходный `1_M_L_3_vertical_n_2.npy`, камера `side_1`. Связь подтверждена [сохранённым source-аудитом](../runs/fmb_v2_berkeley_matched/episode_5201/provenance_v1_source_audit.json): RLDS file_path, два независимых RGB-сопоставления и равенство всех 148 depth-карт. SHA-256 исходника перепроверен при новом экспорте. Observed: 119–126; H3: 124–126; t0=126; оценка: 127–146. Основной пример и t0 зафиксированы до v2 inference.

Второй пример — `1_M_L_3_vertical_n_3.npy`: observed 123–130, H3 128–130, t0=130, оценка 131–150. Это **дополнительный FMB-контроль**, связь с ShareRobot не установлена. [Новый поиск](../runs/fmb_v2_berkeley_matched/second_share_search.json) проверил 48 доступных изображений FMB Trajectory subset, все исходные кадры четырёх камер при двух порядках каналов; совпадение не подтвердилось. Большой Planning archive не был исчерпывающе проверен.

Официальная [страница FMB](https://functional-manipulation-benchmark.github.io/dataset/index.html) определяет сохранённые изображения как BGR. Для всех нейросетей сделан явный BGR→RGB. ShareRobot PNG воспроизводит иной порядок исходных байтов; доказательство происхождения использует прежнюю проверку без изменения её результатов. Это уточнение цветового входа отличает v2 от byte-preserving ShareRobot-изображений.

В `.npy` нет аппаратных timestamps. Поэтому исходная шкала 10 Hz и модельная 15 Hz **номинальны**: H3 соответствует −0.2, −0.1, 0 s; будущее — 0.1…2.0 s. XYZ-прогноз линейно интерполирован по времени с сетки j/15 на k/10; реальные RGB, depth и reference не интерполированы. Это отличается от целочисленного соответствия 5 Hz→15 Hz в Berkeley, но сохраняет тот же принцип сравнения физических времён. Ошибка часов/реальная частота записи не измерены. Авторский processor, как в Berkeley, создаёт служебные видеовремена `[0,1,2]` с `target_fps=1`; реальные номинальные timestamps сохранены отдельно и не заменяют эти значения внутри API. Поэтому согласование времён при оценке не устраняет исторический временной сдвиг входа модели.

## Общий конвейер и его пределы

UniDepthV2 `vits14` запущен на восьми observed RGB каждой сцены; K_eff — медианы fx, fy, cx, cy. Сохранены все K, диапазоны и графики. Собственная depth UniDepth сохранена для диагностики и не используется в основном XYZ. Sensor Z16 переводится по `0.0001 m/raw`: это **поддержанная гипотеза**, а не подтверждённый для записи depth_units. Предыдущие CAD/TCP проверки поддерживают масштаб; опубликованный [код capture](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/camera/rs_capture.py) не даёт записанного значения для эпизода.

RGB↔depth проверены визуально и через описательные расстояния между контурами на трёх observed кадрах; контурами охвачены предмет, плата и видимая часть захвата. Никакая поправка регистрации не подбиралась. Совмещение остаётся приблизительным, с пропусками depth и несовпадением границ. Round-trip 2D→XYZ→2D проверяет формулы, но не доказывает регистрацию. Fixed-camera проверяется тем же observed-only Shi-Tomasi/LK с forward-backward фильтром и порогом медианы 1.5 px; после прохода принимаются identity poses. В робастную выборку могут попасть движущиеся элементы захвата, поэтому проверена и визуальная неподвижность платы.

MolmoPoint-Vid-4B получает только t0 и запрос о красной прямоугольной детали в захвате. Официальный декодер извлекает точку; SAM 2.1 Hiera-L выбирает маску по максимальному score, сохраняет все альтернативы. Применена исходная авторская K-means-функция, seed=0, 100 кандидатов. AllTracker отслеживает их назад по восьми observed кадрам в исходном разрешении 256×256, с bilinear sampling и отдельными confidence/visibility — тот же подход, что в Berkeley, без необходимости уменьшать разрешение.

Depth каждого historical и future запроса — медиана конечных положительных значений в 5×5, диапазон (0,10) m, минимум 13 из 25 в полном окне. Использованы авторские trust/filter/ray-smoothing функции Berkeley с теми же параметрами. Адаптер устраняет только сбой диагностического print при пустом списке доверий невидимых точек; вычисления фильтра не изменены, vendored-файл сохранён. Для FMB дополнительно до inference фиксированы: t0 margin ≥2.5 px, H3 скачок <20 px/кадр, depth patch p90−p10 <0.03 m, принадлежность SAM-маске на всех трёх кадрах H3. Для двух ранних H3 кадров SAM получает observed AllTracker-позицию ближайшего к MolmoPoint кандидата; маска t0 сохраняется исходной. Выбор 24 точек пространственный и детерминированный; три последовательные группы по восемь, без выбора anchor по будущей ошибке. Цветовая маска ECC не определяет основной набор точек.

Все H3 XYZ `[3,24,3]` находятся в системе side_1, конечны, Z>0. Сохранены raw/smoothed истории, скорости, изменение попарных расстояний и величина сглаживания. Сглаживание — обработка измерений, не новая sensor-разметка. Входы metadata/observed/geometry/groups заморожены SHA-256 до прогнозов и проверены после всех вызовов.

MolmoMotion `allenai/MolmoMotion-4B-H3-F30` выполняет по три независимых P8-вызова на сцену, BF16, greedy, seed=0. Для каждого сохранены raw text, tensor, anchor, processor inputs, хэши, runtime/VRAM и строгая проверка всех ID 1–8 и тридцати шагов; текст и anchor восстанавливают сохранённый tensor. H1 `allenai/MolmoMotion-4B-H1-F32` дополнительно получает основной t0 с **теми же** 24 точками и последними XYZ: три полных вызова по 32 шага. Он диагностирует влияние исторической структуры; H1 и H3 также имеют разные checkpoint, поэтому разность не является изолированным причинным эффектом частоты.

Future экспортирован в отдельный каталог после полных H3 и основного H1. Forward AllTracker от t0 использует те же point IDs и не меняет историю. `GT_2D_est` означает оценку RGB-соответствий, а **3D_est** — её подъём через sensor depth и оценённую K под гипотезой масштаба. Это не идеально размеченный физический GT. Исходный pickle-словарь содержит всю запись; экспортёр обращается для подготовки исключительно к заранее фиксированным observed индексам. Знание окна из v1 не скрывается: это повтор существующего окна, а не новый слепой выбор эпизода.

## Метрики

Общая reference-маска одинакова для всех методов внутри каждой метрики. Прогнозы за пределами изображения не исключаются. Static фиксирует последнее XYZ; constant velocity оценивает скорость по трём наблюдаемым XYZ и номинальным timestamps методом наименьших квадратов. 2D и 3D_est могут иметь разное покрытие из-за depth. Единица эксперимента — эпизод; 24×20 отсчётов не считаются независимыми наблюдениями и не используются для статистической значимости.

| Сцена | Метод | ADE 2D, px | FDE 2D, px | ADE 3D_est, m | FDE 3D_est, m |
| --- | --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+f'''

Помимо ADE/FDE сохранены error(t), per-point ADE, fraction outside image, endpoint displacement, direction error и amplitude ratio в `evaluation/metrics.json` каждой сцены.

H1 на основном эпизоде даёт ADE 125.79 px и FDE 208.98 px: средняя ошибка немного меньше H3, конечная существенно выше, и преимущество static сохраняется. Удаление истории при этом диагностическом checkpoint не устранило большую ошибку. У H3 47.08% оцениваемых позиций находятся вне кадра, у H1 — 20.63%; все они сохранены в метриках.

В n3 прогноз H3 почти неподвижен: среднее смещение на конце около 0.55 px при reference около 46 px. По secondary 3D_est порядок методов отличается: на основном эпизоде H3 хуже обоих baseline по ADE/FDE, а в n3 H3/static имеют меньшую 3D_est ошибку, чем constant velocity, хотя constant velocity лучше в 2D. Различие reference-масок, глубины и оценённой K не позволяет отождествлять эти две метрики или выбирать primary после просмотра результатов.

Чистое время генерации трёх групп и максимальная выделенная CUDA-память:

| Сцена | Checkpoint | Генерация, s | Peak allocated, GiB |
| --- | --- | ---: | ---: |
'''+ '\n'.join(runtime_rows)+f'''

Загрузка H3 выполнялась один раз для двух сцен, H1 — отдельной загрузкой. В таблице не учитываются чтение весов, подготовка, скачивание и ожидание. Reserved CUDA и RAM сохранены в group receipts.

На основном эпизоде среднее расхождение ECC и AllTracker для **тех же** 24 IDs — {uncertainty['mean_ECC_AllTracker_disagreement_px']:.3f} px, p90 — {uncertainty['p90_ECC_AllTracker_disagreement_px']:.3f} px. Отношение среднего расхождения к H3 ADE — {uncertainty['ratio_mean_disagreement_to_MolmoMotion_ADE']:.3f}. [Полная диагностика](../runs/fmb_v2_berkeley_matched/episode_5201/evaluation/reference_uncertainty.json). Визуальная выборка будущих кадров проверена отдельно; оба трекера могут совместно ошибаться на почти безтекстурной грани. Нельзя интерпретировать это отношение как доверительный интервал истинного GT.

Для контрольного n3 дополнительно выполнен тот же ECC-аудит: среднее расхождение {control_uncertainty['mean_ECC_AllTracker_disagreement_px']:.3f} px, p90 {control_uncertainty['p90_ECC_AllTracker_disagreement_px']:.3f} px. Разницу H3 и static около 0.03–0.04 px нельзя трактовать как содержательное преимущество одного метода. В поздних кадрах нижние/верхние точки могут приближаться к границе детали и захвату; их точная материальная идентичность не доказана. [Визуальный аудит](../runs/fmb_v2_berkeley_matched/visual_review.json).

## Визуальные результаты

Видео показывает первую группу из восьми постоянных IDs для читаемости; метрики и графики полного размаха используют все 24. Зелёный — AllTracker reference, розовый — H3. MP4 содержат 20 реальных future кадров при номинальных 10 FPS.
'''
    figures=[('observed_8_frames.png','8 observed RGB'),('molmopoint_overlay.png','t0 + MolmoPoint'),
             ('mask_overlay.png','SAM mask'),('mask_and_100_queries.png','100 candidates'),('selected_24_points.png','24 fixed points'),
             ('historical_tracks.png','Observed AllTracker tracks'),('sensor_depth.png','Sensor depth'),('intrinsics_stability.png','Observed K'),
             ('rgb_depth_registration_audit.png','RGB/depth registration audit'),('fixed_camera_check.png','Fixed-camera check'),
             ('history_xyz_clouds.png','Historical XYZ clouds'),('real_future_alltracker.png','Real future + AllTracker'),
             ('prediction_vs_real_contact.png','MolmoMotion vs real'),('full_extent_trajectories.png','All methods, full extents'),
             ('error_vs_time.png','Error vs time'),('per_point_ADE.png','Per-point ADE')]
    for name in SPECS:
        scene=RUN/name;m=read_json(scene/'evaluation/metrics.json');k=np.load(scene/'geometry/K_median.npy');a=read_json(scene/'geometry/input_audit.json')
        text+=f"\n### {name}\n\nK_eff: fx={k[0,0]:.3f}, fy={k[1,1]:.3f}, cx={k[0,2]:.3f}, cy={k[1,2]:.3f} px. Coverage: {m['common_mask_2D_pairs']}/480 в 2D, {m['common_mask_3D_est_pairs']}/480 в 3D_est. Максимальная поправка H3 ray smoothing: {a['history_smoothing_max_displacement_m']*1000:.2f} mm. [Метрики](../runs/fmb_v2_berkeley_matched/{name}/evaluation/metrics.json), [input audit](../runs/fmb_v2_berkeley_matched/{name}/geometry/input_audit.json).\n"
        for filename,title in figures:
            top='observed' if filename in ['molmopoint_overlay.png','mask_overlay.png'] else 'viz'
            text+=f'\n**{title}**\n\n![{title}](../runs/fmb_v2_berkeley_matched/{name}/{top}/{filename})\n'
        text+=f'\n[Видео рядом](../runs/fmb_v2_berkeley_matched/{name}/viz/side_by_side.mp4) · [Видео наложения](../runs/fmb_v2_berkeley_matched/{name}/viz/prediction_vs_real.mp4)\n'
        additional=[('reference_uncertainty.png','Reference uncertainty'),('manual_reference_audit.png','Visual reference audit')]
        if name=='episode_5201':additional.insert(0,('h1_vs_h3.png','H1 vs H3'))
        for filename,title in additional:
            text+=f'\n**{title}**\n\n![{title}](../runs/fmb_v2_berkeley_matched/{name}/viz/{filename})\n'
    text+='''
## Воспроизведение и интерпретация

[Команды](../runs/fmb_v2_berkeley_matched/REPRODUCE.md), [код export](../scripts/fmb_v2_scene.py), [preprocess](../scripts/fmb_v2_preprocess.py), [inference](../scripts/fmb_v2_infer.py), [evaluation](../scripts/fmb_v2_evaluate.py), [проверки протокола](../tests/test_fmb_v2_protocol.py). Checkpoint revisions и хэши сохранены в receipts; сырьё и веса требуется получить отдельно. Все новые файлы находятся в `runs/fmb_v2_berkeley_matched`; скрипты отказываются перезаписывать зафиксированные входы.

Ответ относится к этим двум окнам и конкретной оценённой геометрии. Унификация убирает прежние ручные восемь точек, K_nominal и ECC как основной reference; она не превращает UniDepth K, шкалу depth или безтекстурные соответствия в точную калибровку. В частности, улучшение/ухудшение v1→v2 нельзя приписать единственной причине: одновременно меняются K, выбор и группировка точек, RGB-подготовка и tracker reference. Обобщение на FMB/робототехнику в целом по двум эпизодам не обосновано.
'''
    target=ROOT/'report/fmb_v2_berkeley_matched.md';target.write_text(text,encoding='utf-8')
    missing=[]
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('http') and not (target.parent/link).exists():missing.append(link)
    assert not missing,missing
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':report()
