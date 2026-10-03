# FMB wrist: ViPE, RealSense и robot TCP

Контролируемый эксперимент с движущимися wrist-камерами и тремя источниками геометрии. Качество изображений и траекторий оценивается встроенным просмотром ассистента, без отдельной VLM с Hugging Face. Абсолютная геометрия двух wrist-view не согласовалась; primary C — принятый observed контроль внутри одной камеры с неопределёнными extrinsics/scale.

[Новые видео: галерея](../runs/fmb_wrist_v3/index.html) · [Протокол](../runs/fmb_wrist_v3/PROTOCOL.md) · [Воспроизведение](../runs/fmb_wrist_v3/REPRODUCE.md) · [Pinned sources](../runs/fmb_wrist_v3/sources/source_receipts.json)

## Основной результат

На wrist_2 primary C почти совпадает со Static: ADE3D_est 42.74 против 42.84mm. Выигрыш 0.11mm не подтверждает полезное улучшение прогноза движения. C лучше CV по estimated3D (74.94mm), но хуже по2D (109.93 против 28.49px).

Ветка A имеет низкую2D ошибку, но её исходное3D расхождение с общим reference равно 590.76mm. После вычитания исходного положения displacement-only ADE 46.54mm также хуже Static; кажущаяся2D близость не подтверждает физическую корректность.

Независимый прогноз wrist_1 C заметно хуже Static и CV: ADE3D_est 193.20/40.95/56.72mm. Траектории разных групп расходятся; улучшение геометрии не обеспечило устойчивый результат между ракурсами.

Сами reference-траектории относительного перемещения двух wrists близки (median3.34mm, endpoint4.76mm), хотя абсолютные облака расходятся примерно на15–20cm. Поэтому локальное движение удаётся проверить лучше абсолютной hand-eye геометрии. Результат ограничен одним эпизодом и estimated reference; физическая калибровка не подтверждена.

## Эпизод и время

Source-matched ShareRobot episode5201, raw FMB `1_M_L_3_vertical_n_2.npy`, t0=126. ViPE/hand-eye/history tracking получают 50 разных observed кадров77–126, nominal10Hz. MolmoMotion получает только124–126. Future127–146 открыто после заморозки входов и успешных прогнозов. Trusted NPY десериализуется целиком; future pixels/depth/poses не используются в подготовке.

Много необычных предметов не требуется. Для методов важнее видимость цели, depth support, статическая текстура фона, перекрытия и multi-axis robot motion. Деталь крупная, доска с отверстиями даёт фоновые признаки; поверхность красной детали малотекстурная. Эпизод исследовался в v1/v2, поэтому новый pipeline не означает новый слепой выбор сцены.

Десять локальных кандидатов просмотрены и проверены RGBD/TCP hand-eye по обеим wrist-камерам. Ни один не дал устойчивые extrinsics по bootstrap. `vertical_n_5` имеет немного лучший диагностический score, но тоже широкий разброс X; matched5201 сохранён для сравнения с прежними исследованиями. У более сильно вращающегося `horizontal_n_0` wrist_1 reprojection gate не прошёл; этот observed контроль сохранён без дополнительных model calls.

[Геометрическое ранжирование](../runs/fmb_wrist_v3/selection/geometry_ranking.json) · [Depth support](../runs/fmb_wrist_v3/selection/target_depth_support.json) · [Provenance5201](../runs/fmb_wrist_v3/source_episode_5201_provenance.json)

![Кандидаты, четыре камеры](../runs/fmb_wrist_v3/selection/all_candidates.png)

## Официальные K и aspect-контроль

Официальный rectified профиль640×480: wrist_2 fx=fy=380.405, cx=315.233, cy=240.445; wrist_1 fx=fy=381.357, cx=310.574, cy=242.832. Capture выбирает RGB640×480; `_get_im` делает cv2.resize до256×256 без crop. Prior K256 масштабируется анизотропно с pixel-center mapping. Factory rectified profile не подтверждает автоматически фактический RGB K, serial или depth_units данной записи.

[Официальный wrist_2](https://github.com/functional-manipulation-benchmark/functional-manipulation-benchmark.github.io/blob/b76d9a9477ee52bf491c40651b2a10aa1076f345/static/files/wrist_2) · [wrist_1](https://github.com/functional-manipulation-benchmark/functional-manipulation-benchmark.github.io/blob/b76d9a9477ee52bf491c40651b2a10aa1076f345/static/files/wrist_1) · [Capture](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/camera/rs_capture.py) · [Resize](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/envs/franka_fmb_env.py)

На каждой wrist-камере выполнены native default ViPE для исходного256×256 и дополнительный default прогон после восстановления4:3 до256×192. В последнем K/depth возвращены к256×256. Это явный preprocessing control, а не побитовое повторение исходного square input. Оригинальный ViPE и его ошибки сохранены. FPS остаётся10, кадры не дублируются; модель всегда видит исходный RGB. Выбор aspect сделан по capture evidence до forecast/future, а не по future ADE.

## Три ветки

| Ветка | Depth | K | Pose |
|---|---|---|---|
| A | ViPE | ViPE | ViPE |
| B | sensor Z16×0.0001 | ViPE | ViPE |
| C | sensor Z16×0.0001 | official prior→256² | TCP×estimated X |

Каждый history XYZ преобразован из camera_t через world в camera_t0. Ветки используют одинаковые24 ID, фиксированный порядок трёх групп8 и авторские trust filtering/ray smoothing. Roundtrip с moving camera проверяет реализацию; он не доказывает RGB-depth регистрацию.

Первый набор100 author KMeans точек на всей SAM-маске дал только13 общих годных tracks wrist_2. До inference повторён тот же KMeans seed0 на SAM interior с sensor/ViPE local-depth support наt0, затем AllTracker и прежние фильтры. Исходный набор сохранён. Итог wrist_2 сосредоточен на верхней грани:22 точки наверху и2 на blade. Wrist_1 покрывает blade шире.

AX=XB использует observed TCP xyz+xyzw и ViPE c2w: первые40 кадров для fit, последние10 для проверки. Square wrist_2 |t_X|≈0.669m превышает gate0.5m. Aspect |t_X|≈0.496m, но heldout static reprojection всё равно не проходит. C использует дополнительный independent RGBD/TCP X: official K, sensor depth и static observed matches, с disjoint heldout suffix. Решение зафиксировано до прогнозов. Хорошая локальная reprojection не подтверждает измеренную hand-eye калибровку.

## wrist_2

| Observed control | Value |
|---|---:|
| Native square AX=XB translation norm | 0.669m |
| Aspect AX=XB translation norm; condition | 0.496m; 6.41 |
| AX=XB heldout median/p90 translation | 109.26/128.20mm |
| AX=XB heldout median/p90 rotation | 21.02/22.60° |
| Independent X heldout median/p90 reprojection | 1.39/3.23px |
| Independent X bootstrap translation spread | 18.8–447.3mm |
| Bootstrap observed relative-path p90 spread | 2.2–117.2mm |

| Geometry | Static median/p90 px | H3 rigidity variation | Local gate |
|---|---:|---:|---|
| A_vipe_full | 4.36/14.51 | 0.047 | False |
| B_sensor_vipe | 10.09/38.19 | 0.087 | False |
| C_sensor_tcp_official | 0.80/3.06 | 0.089 | True |

Frozen primary **C_sensor_tcp_official**. Global physical certification: **False**. A/B — диагностические controls; wrist_1 повторяет C.

| K256 | fx | fy | cx | cy |
|---|---:|---:|---:|---:|
| official prior | 152.162 | 202.883 | 125.793 | 128.004 |
| ViPE aspect | 161.928 | 215.904 | 128.000 | 128.167 |
| UniDepth median | 218.720 | 211.931 | 128.350 | 127.912 |

Depth comparison наt0:

| Region | Valid px | median/p90 absΔZ m | ViPE/sensor |
|---|---:|---:|---:|
| target | 2211 | 0.62/0.99 | 5.94 |
| robot_approximate | 6090 | 0.67/1.13 | 4.66 |
| static_table_bin_approximate | 19078 | 0.78/0.95 | 3.94 |
| background | 18724 | 0.68/0.95 | 2.82 |

[Все depth regions и temporal std](../runs/fmb_wrist_v3/wrist_2/geometry/depth_comparison.json) · [Bootstrap X](../runs/fmb_wrist_v3/wrist_2/geometry/independent_robot_rgbd_bootstrap.json) · [Square ViPE control](../runs/fmb_wrist_v3/wrist_2/geometry/native_square_control.json)

Robot/background regions обозначены грубо по RGB. Temporal std региональных медиан включает смену поверхности и camera motion; это описательная вариация, не чистый sensor noise.

![Observed RGB](../runs/fmb_wrist_v3/wrist_2/viz/observed_prefix_review.png)

![RGB-depth edges](../runs/fmb_wrist_v3/wrist_2/viz/registration_review.png)

![100→24 points](../runs/fmb_wrist_v3/wrist_2/viz/candidates_to_selected.png)

![Historical IDs](../runs/fmb_wrist_v3/wrist_2/viz/selected_history_tracks.png)

![K across observed frames](../runs/fmb_wrist_v3/wrist_2/viz/intrinsics_observed_frames.png)

![Pose and AX=XB residuals](../runs/fmb_wrist_v3/wrist_2/viz/pose_K_comparison.png)

![3D camera paths](../runs/fmb_wrist_v3/wrist_2/viz/camera_paths_3d.png)

![Sensor vs ViPE depth](../runs/fmb_wrist_v3/wrist_2/viz/depth_comparison.png)

![History in camera_t0](../runs/fmb_wrist_v3/wrist_2/viz/history_3d_clouds.png)

![molmopoint_overlay.png](../runs/fmb_wrist_v3/wrist_2/observed/molmopoint_overlay.png)

![mask_overlay.png](../runs/fmb_wrist_v3/wrist_2/observed/mask_overlay.png)

### Forecasts and baselines

| Method | ADE/FDE 3D_est mm | ADE/FDE 2D px | Outside | Amplitude ratio |
|---|---:|---:|---:|---:|
| C_sensor_tcp_official | 42.74/88.83 | 109.93/349.54 | 61.3% | 0.01 |
| C_sensor_tcp_official/Static | 42.84/88.89 | 110.50/351.29 | 61.3% | 0.00 |
| C_sensor_tcp_official/CV | 74.94/139.67 | 28.49/35.39 | 0.0% | 2.29 |
| A_vipe_full | 567.75/547.48 | 10.62/25.09 | 0.0% | 0.04 |
| A_vipe_full/Static | 572.81/552.58 | 10.58/24.94 | 0.0% | 0.00 |
| A_vipe_full/CV | 621.17/645.90 | 10.65/23.40 | 0.0% | 1.08 |
| B_sensor_vipe | 42.02/87.53 | 103.54/336.64 | 59.8% | 0.01 |
| B_sensor_vipe/Static | 41.97/87.52 | 103.55/336.41 | 59.8% | 0.00 |
| B_sensor_vipe/CV | 56.87/114.07 | 352.48/2531.66 | 60.8% | 0.51 |

| Geometry forecast | Initial reference offset mm | Displacement-only ADE/FDE mm |
|---|---:|---:|
| C_sensor_tcp_official | 0.41 | 42.66/88.76 |
| A_vipe_full | 590.76 | 46.54/92.04 |
| B_sensor_vipe | 3.47 | 42.81/88.83 |

Absolute ADE3D_est остаётся основной метрикой и включает исходный geometry offset. Displacement-only диагностика вычитает собственныйt0 каждого forecast и reference, чтобы отдельно показать ошибку движения. Она не исправляет scale/pose uncertainty.

Common masks: 449/480 3D и 480/480 2D samples. Off-image/behind-camera forecasts сохраняются в numerical errors. Static/CV используют тот же H3 и геометрию своей ветки.

Единый reference всех веток: future AllTracker + sensor5×5 median + official K + future TCP×frozen observed X. Future poses используются только в оценке; XYZ15Hz интерполируется по времени к nominal10Hz. Проекция выполняется в каждой движущейся future camera. Это **3D_est**, не физический GT.

[Метрики](../runs/fmb_wrist_v3/wrist_2/evaluation/metrics.json) · [Side-by-side видео](../runs/fmb_wrist_v3/wrist_2/viz/geometry_forecasts_side_by_side.mp4) · [Primary vs real](../runs/fmb_wrist_v3/wrist_2/viz/primary_prediction_vs_real.mp4)

Observed bootstrap X меняет будущий reference: maxp90 13.6mm. Это sensitivity envelope, не confidence interval; future ничего не подгоняет.

| Geometry | Bootstrap fits: model ADE ниже Static | Ниже CV |
|---|---:|---:|
| A_vipe_full | 7/7 | 7/7 |
| B_sensor_vipe | 0/7 | 7/7 |
| C_sensor_tcp_official | 7/7 | 7/7 |

Эти counts описывают чувствительность к reference X при фиксированных model inputs; они не являются статистической значимостью или независимыми эпизодами.

![Real future + AllTracker24](../runs/fmb_wrist_v3/wrist_2/viz/reference_all24_review.png)

![Methods hidden for first visual assessment](../runs/fmb_wrist_v3/wrist_2/viz/blinded_review_20.png)

![Forecast vs real at2s](../runs/fmb_wrist_v3/wrist_2/viz/forecast_review_20.png)

![Error vs time](../runs/fmb_wrist_v3/wrist_2/viz/forecast_errors.png)

![Full30-step range](../runs/fmb_wrist_v3/wrist_2/viz/full_30step_xyz.png)

![Hand-eye reference sensitivity](../runs/fmb_wrist_v3/wrist_2/viz/reference_hand_eye_sensitivity.png)

## wrist_1

| Observed control | Value |
|---|---:|
| Native square AX=XB translation norm | 0.893m |
| Aspect AX=XB translation norm; condition | 0.514m; 6.41 |
| AX=XB heldout median/p90 translation | 29.99/89.19mm |
| AX=XB heldout median/p90 rotation | 21.41/29.52° |
| Independent X heldout median/p90 reprojection | 1.29/2.62px |
| Independent X bootstrap translation spread | 25.8–739.7mm |
| Bootstrap observed relative-path p90 spread | 9.0–333.7mm |

| Geometry | Static median/p90 px | H3 rigidity variation | Local gate |
|---|---:|---:|---|
| A_vipe_full | 2.46/7.05 | 0.039 | True |
| B_sensor_vipe | 20.68/52.32 | 0.129 | False |
| C_sensor_tcp_official | 1.29/3.10 | 0.077 | True |

Frozen primary **C_sensor_tcp_official**. Global physical certification: **False**. A/B — диагностические controls; wrist_1 повторяет C.

| K256 | fx | fy | cx | cy |
|---|---:|---:|---:|---:|
| official prior | 152.543 | 203.390 | 123.930 | 129.277 |
| ViPE aspect | 421.059 | 561.411 | 128.000 | 128.167 |
| UniDepth median | 248.890 | 244.405 | 128.350 | 127.787 |

Depth comparison наt0:

| Region | Valid px | median/p90 absΔZ m | ViPE/sensor |
|---|---:|---:|---:|
| target | 2807 | 0.64/0.81 | 6.47 |
| robot_approximate | 7888 | 0.79/0.93 | 3.68 |
| static_table_bin_approximate | 20635 | 0.76/0.88 | 3.52 |
| background | 15290 | 0.69/0.86 | 2.80 |

[Все depth regions и temporal std](../runs/fmb_wrist_v3/wrist_1/geometry/depth_comparison.json) · [Bootstrap X](../runs/fmb_wrist_v3/wrist_1/geometry/independent_robot_rgbd_bootstrap.json) · [Square ViPE control](../runs/fmb_wrist_v3/wrist_1/geometry/native_square_control.json)

Robot/background regions обозначены грубо по RGB. Temporal std региональных медиан включает смену поверхности и camera motion; это описательная вариация, не чистый sensor noise.

![Observed RGB](../runs/fmb_wrist_v3/wrist_1/viz/observed_prefix_review.png)

![RGB-depth edges](../runs/fmb_wrist_v3/wrist_1/viz/registration_review.png)

![100→24 points](../runs/fmb_wrist_v3/wrist_1/viz/candidates_to_selected.png)

![Historical IDs](../runs/fmb_wrist_v3/wrist_1/viz/selected_history_tracks.png)

![K across observed frames](../runs/fmb_wrist_v3/wrist_1/viz/intrinsics_observed_frames.png)

![Pose and AX=XB residuals](../runs/fmb_wrist_v3/wrist_1/viz/pose_K_comparison.png)

![3D camera paths](../runs/fmb_wrist_v3/wrist_1/viz/camera_paths_3d.png)

![Sensor vs ViPE depth](../runs/fmb_wrist_v3/wrist_1/viz/depth_comparison.png)

![History in camera_t0](../runs/fmb_wrist_v3/wrist_1/viz/history_3d_clouds.png)

![molmopoint_overlay.png](../runs/fmb_wrist_v3/wrist_1/observed/molmopoint_overlay.png)

![mask_overlay.png](../runs/fmb_wrist_v3/wrist_1/observed/mask_overlay.png)

### Forecasts and baselines

| Method | ADE/FDE 3D_est mm | ADE/FDE 2D px | Outside | Amplitude ratio |
|---|---:|---:|---:|---:|
| C_sensor_tcp_official | 193.20/343.00 | 1355.61/1597.53 | 57.1% | 3.20 |
| C_sensor_tcp_official/Static | 40.95/89.66 | 102.66/342.15 | 45.4% | 0.00 |
| C_sensor_tcp_official/CV | 56.72/100.18 | 76.26/137.05 | 2.7% | 1.86 |

| Geometry forecast | Initial reference offset mm | Displacement-only ADE/FDE mm |
|---|---:|---:|
| C_sensor_tcp_official | 0.67 | 193.39/343.23 |

Absolute ADE3D_est остаётся основной метрикой и включает исходный geometry offset. Displacement-only диагностика вычитает собственныйt0 каждого forecast и reference, чтобы отдельно показать ошибку движения. Она не исправляет scale/pose uncertainty.

Common masks: 420/480 3D и 480/480 2D samples. Off-image/behind-camera forecasts сохраняются в numerical errors. Static/CV используют тот же H3 и геометрию своей ветки.

Единый reference всех веток: future AllTracker + sensor5×5 median + official K + future TCP×frozen observed X. Future poses используются только в оценке; XYZ15Hz интерполируется по времени к nominal10Hz. Проекция выполняется в каждой движущейся future camera. Это **3D_est**, не физический GT.

[Метрики](../runs/fmb_wrist_v3/wrist_1/evaluation/metrics.json) · [Side-by-side видео](../runs/fmb_wrist_v3/wrist_1/viz/geometry_forecasts_side_by_side.mp4) · [Primary vs real](../runs/fmb_wrist_v3/wrist_1/viz/primary_prediction_vs_real.mp4)

Observed bootstrap X меняет будущий reference: maxp90 26.0mm. Это sensitivity envelope, не confidence interval; future ничего не подгоняет.

| Geometry | Bootstrap fits: model ADE ниже Static | Ниже CV |
|---|---:|---:|
| C_sensor_tcp_official | 0/7 | 0/7 |

Эти counts описывают чувствительность к reference X при фиксированных model inputs; они не являются статистической значимостью или независимыми эпизодами.

![Real future + AllTracker24](../runs/fmb_wrist_v3/wrist_1/viz/reference_all24_review.png)

![Methods hidden for first visual assessment](../runs/fmb_wrist_v3/wrist_1/viz/blinded_review_20.png)

![Forecast vs real at2s](../runs/fmb_wrist_v3/wrist_1/viz/forecast_review_20.png)

![Error vs time](../runs/fmb_wrist_v3/wrist_1/viz/forecast_errors.png)

![Full30-step range](../runs/fmb_wrist_v3/wrist_1/viz/full_30step_xyz.png)

![Hand-eye reference sensitivity](../runs/fmb_wrist_v3/wrist_1/viz/reference_hand_eye_sensitivity.png)

## Cross-view controls

Observed wrist board nearest-surface median/p90 disagreement 154.0/185.2mm; target centroid 196.1mm. Разные видимые поверхности не объясняют уверенно такую ошибку, поэтому абсолютная геометрия не подтверждена.

Side_1 static depth-supported matches:5, side_2:0; base alignment не принимается. Side controls используют own-camera sensor XYZ displacement видимой цветовой поверхности. Частичные контуры, surface bias и разные point identities ограничивают сравнение.

![Observed wrist clouds](../runs/fmb_wrist_v3/wrist_crossview_base_clouds.png)

[Observed four-view audit](../runs/fmb_wrist_v3/crossview_observed_audit.json)

Estimated world-frame displacement disagreement wrist1↔wrist2: median 3.3mm, endpoint 4.8mm. Наборы точек на разных поверхностях независимо выбраны; rotation и X uncertainty влияют на это сравнение.

![Wrist world motion replication](../runs/fmb_wrist_v3/wrist_future_world_motion.png)

| Side | Endpoint centroid displacement mm | Median magnitude disagreement mm |
|---|---:|---:|
| side_1 | 82.61 | 4.99 |
| side_2 | 23.19 | 22.76 |

Side_1 поддерживает приблизительную величину движения, но его visible centroid не совпадает с wrist material points. Side_2 обрезает объект верхней границей во всех21 кадре: его численное displacement не считается надёжной проверкой движения целого объекта.

![Side/wrist motion magnitude](../runs/fmb_wrist_v3/side_future_motion_crosscheck.png)

![side_1 partial contour control](../runs/fmb_wrist_v3/side_1/viz/side_future_color_control.png)

![side_2 partial contour control](../runs/fmb_wrist_v3/side_2/viz/side_future_color_control.png)

## Встроенная оценка изображений

Observed review сохранён до модели: image hashes, source IDs и point flags. На первых future sheets методам присвоены скрытые имена Forecast1…; reference drift проверяется отдельно. Первые качественные наблюдения сохранены до раскрытия карты методов и numerical comparison; итоговая интерпретация дополнена после раскрытия метрик. Предыдущие гипотезы о геометрии были известны, поэтому blinding скрывает названия, но не даёт полностью независимую оценку. Это оценка ассистента, не воспроизводимый HF checkpoint или calibrated metric.

[Observed visual review](../runs/fmb_wrist_v3/builtin_visual_review_observed.json) · [Первая blinded future оценка](../runs/fmb_wrist_v3/builtin_blinded_review.json)

Встроенный просмотр подтвердил: на wrist_2 прогнозы A/B/C почти статичны в camera_t0. Для primary C ADE3D_est составляет42.74mm против42.84mm у Static и74.94mm у CV; выигрыш0.11mm у Static практически мал и не является убедительным улучшением. CV лучше в2D:28.49px против109.93px у C. A выглядит ближе к reference в2D, но исходное расхождение3D равно590.76mm, а displacement-only ADE46.54mm хуже Static42.76mm. На wrist_1 C даёт несогласованные между группами траектории и ADE3D_est193.20mm против40.95mm у Static и56.72mm у CV. В reference не обнаружено явного выхода выбранных ID на фон в трёх просмотренных кадрах; низкая текстура и перекрытия не позволяют подтвердить точную материальную идентичность. Две wrist-реконструкции относительного перемещения согласуются гораздо лучше абсолютных облаков (median3.34mm, endpoint4.76mm), side_1 приблизительно поддерживает движение; side_2 сильно обрезает объект и не даёт надёжной motion-проверки. Эти выводы относятся к estimated reference при неопределённых K/X/depth scale, а не к физическому GT.

[Future review и hashes](../runs/fmb_wrist_v3/builtin_visual_review_future.json)

## Проверки и границы вывода

MolmoMotion-4B-H3-F30 pinned3f5e790a…, BF16, greedy, seed0. Wrist_2:9 завершённых P8 calls; wrist_1:3 завершённых C calls. Всего13 начатых attempts,12 полных outputs. Первая дополнительная попытка A/group0 прервана при конкурирующем GPU workload и сохранена отдельно; её повтор использует те же frozen inputs. Вызовы сохраняют actual processor tensors, anchor, input hashes, runtime/VRAM, raw text и strict parser reconstruction всех8×30×3 outputs.

Sensor scale0.0001m/unit остаётся гипотезой без recording-time depth_units. RGB-depth registration и factory↔RGB K неполностью установлены, bootstrap X неустойчив. Камеры получают последний кадр независимых потоков, без hardware timestamps; exact stereo triangulation/four-view fusion не заявляются. Native processor H3 timing отличается от nominal source10Hz; payload сохранён.

Геометрические gate failures — измеренный результат исследования. Сравнение C/Static/CV относится к принятому estimated reference; absolute physical forecast accuracy ограничена калибровкой. Единица эксперимента — один episode;24 paths не являются24 независимыми наблюдениями. FMBv1/v2 сохранены.

[Полный completion audit](../runs/fmb_wrist_v3/completion_audit.json)

