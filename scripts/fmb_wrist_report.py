"""Generate the final research report from actual outputs and built-in reviews."""
import json
import numpy as np
from fmb_wrist_prepare import ROOT,RUN,write,sha

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def fmt(value):return '—' if value is None else f'{value:.2f}'

def build():
    ready=all((RUN/camera/'evaluation/metrics.json').exists() for camera in ['wrist_2','wrist_1'])
    lines=['# FMB wrist: ViPE, RealSense и robot TCP','',
      'Контролируемый эксперимент с движущимися wrist-камерами и тремя источниками геометрии. Качество изображений и траекторий оценивается встроенным просмотром ассистента, без отдельной VLM с Hugging Face. Абсолютная геометрия двух wrist-view не согласовалась; primary C — принятый observed контроль внутри одной камеры с неопределёнными extrinsics/scale.','',
      '[Новые видео: галерея](../runs/fmb_wrist_v3/index.html) · [Протокол](../runs/fmb_wrist_v3/PROTOCOL.md) · [Воспроизведение](../runs/fmb_wrist_v3/REPRODUCE.md) · [Pinned sources](../runs/fmb_wrist_v3/sources/source_receipts.json)','',
      '## Эпизод и время','',
      'Source-matched ShareRobot episode5201, raw FMB `1_M_L_3_vertical_n_2.npy`, t0=126. ViPE/hand-eye/history tracking получают 50 разных observed кадров77–126, nominal10Hz. MolmoMotion получает только124–126. Future127–146 открыто после заморозки входов и успешных прогнозов. Trusted NPY десериализуется целиком; future pixels/depth/poses не используются в подготовке.','',
      'Много необычных предметов не требуется. Для методов важнее видимость цели, depth support, статическая текстура фона, перекрытия и multi-axis robot motion. Деталь крупная, доска с отверстиями даёт фоновые признаки; поверхность красной детали малотекстурная. Эпизод исследовался в v1/v2, поэтому новый pipeline не означает новый слепой выбор сцены.','',
      'Десять локальных кандидатов просмотрены и проверены RGBD/TCP hand-eye по обеим wrist-камерам. Ни один не дал устойчивые extrinsics по bootstrap. `vertical_n_5` имеет немного лучший диагностический score, но тоже широкий разброс X; matched5201 сохранён для сравнения с прежними исследованиями. У более сильно вращающегося `horizontal_n_0` wrist_1 reprojection gate не прошёл; этот observed контроль сохранён без дополнительных model calls.','',
      '[Геометрическое ранжирование](../runs/fmb_wrist_v3/selection/geometry_ranking.json) · [Depth support](../runs/fmb_wrist_v3/selection/target_depth_support.json) · [Provenance5201](../runs/fmb_wrist_v3/source_episode_5201_provenance.json)','',
      '![Кандидаты, четыре камеры](../runs/fmb_wrist_v3/selection/all_candidates.png)','',
      '## Официальные K и aspect-контроль','',
      'Официальный rectified профиль640×480: wrist_2 fx=fy=380.405, cx=315.233, cy=240.445; wrist_1 fx=fy=381.357, cx=310.574, cy=242.832. Capture выбирает RGB640×480; `_get_im` делает cv2.resize до256×256 без crop. Prior K256 масштабируется анизотропно с pixel-center mapping. Factory rectified profile не подтверждает автоматически фактический RGB K, serial или depth_units данной записи.','',
      '[Официальный wrist_2](https://github.com/functional-manipulation-benchmark/functional-manipulation-benchmark.github.io/blob/b76d9a9477ee52bf491c40651b2a10aa1076f345/static/files/wrist_2) · [wrist_1](https://github.com/functional-manipulation-benchmark/functional-manipulation-benchmark.github.io/blob/b76d9a9477ee52bf491c40651b2a10aa1076f345/static/files/wrist_1) · [Capture](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/camera/rs_capture.py) · [Resize](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/envs/franka_fmb_env.py)','',
      'На каждой wrist-камере выполнены native default ViPE для исходного256×256 и дополнительный default прогон после восстановления4:3 до256×192. В последнем K/depth возвращены к256×256. Это явный preprocessing control, а не побитовое повторение исходного square input. Оригинальный ViPE и его ошибки сохранены. FPS остаётся10, кадры не дублируются; модель всегда видит исходный RGB. Выбор aspect сделан по capture evidence до forecast/future, а не по future ADE.','',
      '## Три ветки','',
      '| Ветка | Depth | K | Pose |','|---|---|---|---|','| A | ViPE | ViPE | ViPE |','| B | sensor Z16×0.0001 | ViPE | ViPE |','| C | sensor Z16×0.0001 | official prior→256² | TCP×estimated X |','',
      'Каждый history XYZ преобразован из camera_t через world в camera_t0. Ветки используют одинаковые24 ID, фиксированный порядок трёх групп8 и авторские trust filtering/ray smoothing. Roundtrip с moving camera проверяет реализацию; он не доказывает RGB-depth регистрацию.','',
      'Первый набор100 author KMeans точек на всей SAM-маске дал только13 общих годных tracks wrist_2. До inference повторён тот же KMeans seed0 на SAM interior с sensor/ViPE local-depth support наt0, затем AllTracker и прежние фильтры. Исходный набор сохранён. Итог wrist_2 сосредоточен на верхней грани:22 точки наверху и2 на blade. Wrist_1 покрывает blade шире.','',
      'AX=XB использует observed TCP xyz+xyzw и ViPE c2w: первые40 кадров для fit, последние10 для проверки. Square wrist_2 |t_X|≈0.669m превышает gate0.5m. Aspect |t_X|≈0.496m, но heldout static reprojection всё равно не проходит. C использует дополнительный independent RGBD/TCP X: official K, sensor depth и static observed matches, с disjoint heldout suffix. Решение зафиксировано до прогнозов. Хорошая локальная reprojection не подтверждает измеренную hand-eye калибровку.','']
    if ready:
        w2=read(RUN/'wrist_2/evaluation/metrics.json')['methods'];w1=read(RUN/'wrist_1/evaluation/metrics.json')['methods']
        c='C_sensor_tcp_official';a='A_vipe_full'
        lines[6:6]=['## Основной результат','',
          f'На wrist_2 primary C почти совпадает со Static: ADE3D_est {w2[c]["3D_est_m"]["ADE"]*1000:.2f} против {w2[c+"/Static"]["3D_est_m"]["ADE"]*1000:.2f}mm. Выигрыш {(w2[c+"/Static"]["3D_est_m"]["ADE"]-w2[c]["3D_est_m"]["ADE"])*1000:.2f}mm не подтверждает полезное улучшение прогноза движения. C лучше CV по estimated3D ({w2[c+"/CV"]["3D_est_m"]["ADE"]*1000:.2f}mm), но хуже по2D ({w2[c]["2D_px"]["ADE"]:.2f} против {w2[c+"/CV"]["2D_px"]["ADE"]:.2f}px).','',
          f'Ветка A имеет низкую2D ошибку, но её исходное3D расхождение с общим reference равно {w2[a]["initial_reference_offset_m"]*1000:.2f}mm. После вычитания исходного положения displacement-only ADE {w2[a]["displacement_3D_est_m"]["ADE"]*1000:.2f}mm также хуже Static; кажущаяся2D близость не подтверждает физическую корректность.','',
          f'Независимый прогноз wrist_1 C заметно хуже Static и CV: ADE3D_est {w1[c]["3D_est_m"]["ADE"]*1000:.2f}/{w1[c+"/Static"]["3D_est_m"]["ADE"]*1000:.2f}/{w1[c+"/CV"]["3D_est_m"]["ADE"]*1000:.2f}mm. Траектории разных групп расходятся; улучшение геометрии не обеспечило устойчивый результат между ракурсами.','',
          'Сами reference-траектории относительного перемещения двух wrists близки (median3.34mm, endpoint4.76mm), хотя абсолютные облака расходятся примерно на15–20cm. Поэтому локальное движение удаётся проверить лучше абсолютной hand-eye геометрии. Результат ограничен одним эпизодом и estimated reference; физическая калибровка не подтверждена.','']
    for camera in ['wrist_2','wrist_1']:
        scene=RUN/camera;selection=read(scene/'geometry/branch_selection.json');diag=read(scene/'geometry/geometry_diagnostics.json')
        hand=read(scene/'geometry/hand_eye.json');direct=read(scene/'geometry/independent_robot_rgbd_hand_eye.json');boot=read(scene/'geometry/independent_robot_rgbd_bootstrap.json')['fits'];native=read(scene/'geometry/native_square_control.json')
        lines += [f'## {camera}','', '| Observed control | Value |','|---|---:|',
          f'| Native square AX=XB translation norm | {native["hand_eye"]["estimated_translation_norm_m"]:.3f}m |',
          f'| Aspect AX=XB translation norm; condition | {hand["estimated_translation_norm_m"]:.3f}m; {hand["translation_system_condition"]:.2f} |',
          f'| AX=XB heldout median/p90 translation | {hand["heldout"]["median_translation_mm"]:.2f}/{hand["heldout"]["p90_translation_mm"]:.2f}mm |',
          f'| AX=XB heldout median/p90 rotation | {hand["heldout"]["median_rotation_deg"]:.2f}/{hand["heldout"]["p90_rotation_deg"]:.2f}° |',
          f'| Independent X heldout median/p90 reprojection | {direct["heldout"]["median_px"]:.2f}/{direct["heldout"]["p90_px"]:.2f}px |',
          f'| Independent X bootstrap translation spread | {min(r["translation_difference_mm"] for r in boot):.1f}–{max(r["translation_difference_mm"] for r in boot):.1f}mm |',
          f'| Bootstrap observed relative-path p90 spread | {min(r["observed_relative_path_p90_translation_difference_mm"] for r in boot):.1f}–{max(r["observed_relative_path_p90_translation_difference_mm"] for r in boot):.1f}mm |','',
          '| Geometry | Static median/p90 px | H3 rigidity variation | Local gate |','|---|---:|---:|---|']
        for name,row in selection['branches'].items():
            s=row['static_reprojection'];lines.append(f'| {name} | {s["median_px"]:.2f}/{s["p90_px"]:.2f} | {row["relative_rigidity_median"]:.3f} | {row["validated_geometry_gate"]} |')
        lines += ['',f'Frozen primary **{selection["primary"]}**. Global physical certification: **False**. A/B — диагностические controls; wrist_1 повторяет C.','', '| K256 | fx | fy | cx | cy |','|---|---:|---:|---:|---:|']
        for label,key in [('official prior','official'),('ViPE aspect','vipe_median'),('UniDepth median','unidepth_median')]:
            k=np.array(diag['K_values'][key]);lines.append(f'| {label} | {k[0,0]:.3f} | {k[1,1]:.3f} | {k[0,2]:.3f} | {k[1,2]:.3f} |')
        depth=read(scene/'geometry/depth_comparison.json');lines += ['','Depth comparison наt0:','', '| Region | Valid px | median/p90 absΔZ m | ViPE/sensor |','|---|---:|---:|---:|']
        for row in depth['rows']:
            if row['local_id']==49:lines.append(f'| {row["region"]} | {row["valid_pixels"]} | {fmt(row["median_abs_depth_difference_m"])}/{fmt(row["p90_abs_depth_difference_m"])} | {fmt(row["median_vipe_sensor_scale_ratio"])} |')
        lines += ['',f'[Все depth regions и temporal std](../runs/fmb_wrist_v3/{camera}/geometry/depth_comparison.json) · [Bootstrap X](../runs/fmb_wrist_v3/{camera}/geometry/independent_robot_rgbd_bootstrap.json) · [Square ViPE control](../runs/fmb_wrist_v3/{camera}/geometry/native_square_control.json)','',
          'Robot/background regions обозначены грубо по RGB. Temporal std региональных медиан включает смену поверхности и camera motion; это описательная вариация, не чистый sensor noise.','']
        for image,title in [('observed_prefix_review.png','Observed RGB'),('registration_review.png','RGB-depth edges'),('candidates_to_selected.png','100→24 points'),('selected_history_tracks.png','Historical IDs'),('intrinsics_observed_frames.png','K across observed frames'),('pose_K_comparison.png','Pose and AX=XB residuals'),('camera_paths_3d.png','3D camera paths'),('depth_comparison.png','Sensor vs ViPE depth'),('history_3d_clouds.png','History in camera_t0')]:lines += [f'![{title}](../runs/fmb_wrist_v3/{camera}/viz/{image})','']
        for image in ['molmopoint_overlay.png','mask_overlay.png']:
            if (scene/'observed'/image).exists():lines += [f'![{image}](../runs/fmb_wrist_v3/{camera}/observed/{image})','']
        if not(scene/'evaluation/metrics.json').exists():continue
        m=read(scene/'evaluation/metrics.json');lines += ['### Forecasts and baselines','', '| Method | ADE/FDE 3D_est mm | ADE/FDE 2D px | Outside | Amplitude ratio |','|---|---:|---:|---:|---:|']
        for name,row in m['methods'].items():
            a,c=row['3D_est_m'],row['2D_px'];lines.append(f'| {name} | {fmt(a["ADE"]*1000)}/{fmt(a["FDE"]*1000)} | {fmt(c["ADE"])}/{fmt(c["FDE"])} | {row["outside_image_fraction"]:.1%} | {fmt(row["median_endpoint_amplitude_ratio"])} |')
        lines += ['','| Geometry forecast | Initial reference offset mm | Displacement-only ADE/FDE mm |','|---|---:|---:|']
        for name,row in m['methods'].items():
            if '/' not in name:
                d=row['displacement_3D_est_m'];lines.append(f'| {name} | {row["initial_reference_offset_m"]*1000:.2f} | {d["ADE"]*1000:.2f}/{d["FDE"]*1000:.2f} |')
        lines += ['','Absolute ADE3D_est остаётся основной метрикой и включает исходный geometry offset. Displacement-only диагностика вычитает собственныйt0 каждого forecast и reference, чтобы отдельно показать ошибку движения. Она не исправляет scale/pose uncertainty.']
        lines += ['',f'Common masks: {m["common_valid_3d_samples"]}/480 3D и {m["common_valid_2d_samples"]}/480 2D samples. Off-image/behind-camera forecasts сохраняются в numerical errors. Static/CV используют тот же H3 и геометрию своей ветки.','',
          'Единый reference всех веток: future AllTracker + sensor5×5 median + official K + future TCP×frozen observed X. Future poses используются только в оценке; XYZ15Hz интерполируется по времени к nominal10Hz. Проекция выполняется в каждой движущейся future camera. Это **3D_est**, не физический GT.','',
          f'[Метрики](../runs/fmb_wrist_v3/{camera}/evaluation/metrics.json) · [Side-by-side видео](../runs/fmb_wrist_v3/{camera}/viz/geometry_forecasts_side_by_side.mp4) · [Primary vs real](../runs/fmb_wrist_v3/{camera}/viz/primary_prediction_vs_real.mp4)','']
        sensitivity=read(scene/'evaluation/hand_eye_reference_sensitivity.json')
        lines += [f'Observed bootstrap X меняет будущий reference: maxp90 {sensitivity["max_bootstrap_p90_reference_difference_mm"]:.1f}mm. Это sensitivity envelope, не confidence interval; future ничего не подгоняет.','']
        lines += ['| Geometry | Bootstrap fits: model ADE ниже Static | Ниже CV |','|---|---:|---:|']
        for name,row in sensitivity['ranking_sensitivity'].items():lines.append(f'| {name} | {row["model_lower_ADE_than_static_fits"]}/{row["bootstrap_fits"]} | {row["model_lower_ADE_than_CV_fits"]}/{row["bootstrap_fits"]} |')
        lines += ['','Эти counts описывают чувствительность к reference X при фиксированных model inputs; они не являются статистической значимостью или независимыми эпизодами.','']
        for image,title in [('reference_all24_review.png','Real future + AllTracker24'),('blinded_review_20.png','Methods hidden for first visual assessment'),('forecast_review_20.png','Forecast vs real at2s'),('forecast_errors.png','Error vs time'),('full_30step_xyz.png','Full30-step range'),('reference_hand_eye_sensitivity.png','Hand-eye reference sensitivity')]:lines += [f'![{title}](../runs/fmb_wrist_v3/{camera}/viz/{image})','']
    cross=read(RUN/'crossview_observed_audit.json');a=cross['wrist_geometry_agreement']
    lines += ['## Cross-view controls','',f'Observed wrist board nearest-surface median/p90 disagreement {a["board_bidirectional_nearest_median_mm"]:.1f}/{a["board_bidirectional_nearest_p90_mm"]:.1f}mm; target centroid {a["target_centroid_difference_mm"]:.1f}mm. Разные видимые поверхности не объясняют уверенно такую ошибку, поэтому абсолютная геометрия не подтверждена.','',
      'Side_1 static depth-supported matches:5, side_2:0; base alignment не принимается. Side controls используют own-camera sensor XYZ displacement видимой цветовой поверхности. Частичные контуры, surface bias и разные point identities ограничивают сравнение.','',
      '![Observed wrist clouds](../runs/fmb_wrist_v3/wrist_crossview_base_clouds.png)','', '[Observed four-view audit](../runs/fmb_wrist_v3/crossview_observed_audit.json)','']
    if (RUN/'wrist_future_world_motion.json').exists():
        m=read(RUN/'wrist_future_world_motion.json');lines += [f'Estimated world-frame displacement disagreement wrist1↔wrist2: median {m["median_displacement_vector_disagreement_mm"]:.1f}mm, endpoint {m["endpoint_displacement_vector_disagreement_mm"]:.1f}mm. Наборы точек на разных поверхностях независимо выбраны; rotation и X uncertainty влияют на это сравнение.','', '![Wrist world motion replication](../runs/fmb_wrist_v3/wrist_future_world_motion.png)','']
    if (RUN/'side_future_crosscheck.json').exists():
        side=read(RUN/'side_future_crosscheck.json');lines += ['| Side | Endpoint centroid displacement mm | Median magnitude disagreement mm |','|---|---:|---:|']
        for row in side['rows']:lines.append(f'| {row["camera"]} | {fmt(row["future_endpoint_displacement_m"]*1000) if row["future_endpoint_displacement_m"] is not None else "—"} | {row["median_abs_motion_magnitude_disagreement_with_wrist_m"]*1000:.2f} |')
        lines += ['','Side_1 поддерживает приблизительную величину движения, но его visible centroid не совпадает с wrist material points. Side_2 обрезает объект верхней границей во всех21 кадре: его численное displacement не считается надёжной проверкой движения целого объекта.','', '![Side/wrist motion magnitude](../runs/fmb_wrist_v3/side_future_motion_crosscheck.png)','']
        for camera in ['side_1','side_2']:lines += [f'![{camera} partial contour control](../runs/fmb_wrist_v3/{camera}/viz/side_future_color_control.png)','']
    lines += ['## Встроенная оценка изображений','',
      'Observed review сохранён до модели: image hashes, source IDs и point flags. На первых future sheets методам присвоены скрытые имена Forecast1…; reference drift проверяется отдельно. Первые качественные наблюдения сохранены до раскрытия карты методов и numerical comparison; итоговая интерпретация дополнена после раскрытия метрик. Предыдущие гипотезы о геометрии были известны, поэтому blinding скрывает названия, но не даёт полностью независимую оценку. Это оценка ассистента, не воспроизводимый HF checkpoint или calibrated metric.','', '[Observed visual review](../runs/fmb_wrist_v3/builtin_visual_review_observed.json) · [Первая blinded future оценка](../runs/fmb_wrist_v3/builtin_blinded_review.json)','']
    if (RUN/'builtin_visual_review_future.json').exists():
        review=read(RUN/'builtin_visual_review_future.json');lines += [review['overall_assessment'],'','[Future review и hashes](../runs/fmb_wrist_v3/builtin_visual_review_future.json)','']
    lines += ['## Проверки и границы вывода','',
      ('MolmoMotion-4B-H3-F30 pinned3f5e790a…, BF16, greedy, seed0. Wrist_2:9 завершённых P8 calls; wrist_1:3 завершённых C calls. Всего13 начатых attempts,12 полных outputs. ' if ready else 'Матрица расчёта:9 успешных P8 calls wrist_2 и3 C calls wrist_1; прогнозы ещё не завершены. ')+ 'Первая дополнительная попытка A/group0 прервана при конкурирующем GPU workload и сохранена отдельно; её повтор использует те же frozen inputs. Вызовы сохраняют actual processor tensors, anchor, input hashes, runtime/VRAM, raw text и strict parser reconstruction всех8×30×3 outputs.','',
      'Sensor scale0.0001m/unit остаётся гипотезой без recording-time depth_units. RGB-depth registration и factory↔RGB K неполностью установлены, bootstrap X неустойчив. Камеры получают последний кадр независимых потоков, без hardware timestamps; exact stereo triangulation/four-view fusion не заявляются. Native processor H3 timing отличается от nominal source10Hz; payload сохранён.','',
      'Геометрические gate failures — измеренный результат исследования. Сравнение C/Static/CV относится к принятому estimated reference; absolute physical forecast accuracy ограничена калибровкой. Единица эксперимента — один episode;24 paths не являются24 независимыми наблюдениями. FMBv1/v2 сохранены.','']
    if (RUN/'completion_audit.json').exists():lines += ['[Полный completion audit](../runs/fmb_wrist_v3/completion_audit.json)','']
    report=ROOT/'report/fmb_wrist_v3.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');write(RUN/'requirements_progress.json',{'report_sha256':sha(report),'completion_audit_required':True})
    print('Report written',report,flush=True)

if __name__=='__main__':build()
