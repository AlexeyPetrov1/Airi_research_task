# Dobb·E episode 3651: проверка RGB-only COLMAP калибровки измеренной глубиной

Дата: 2026-10-02. **Итог строгого геометрического gate: `BLOCKED`, `molmo_ready=false`.** Для основной сцены RGB-only COLMAP не дал устойчивую `K`; измеренная depth и опубликованные poses не подтвердили согласованную метрическую реконструкцию. Достоверные полные 3D GT/ADE/FDE не получены. По просьбе пользователя отдельно построены [приближённые `(3,8,3)` и прогноз MolmoMotion](dobbe_approx_colmap_f2nerf_molmo.md) с частичной оценкой; они не меняют формальный gate. Решение записано в [decision.json](../runs/dobbe_rgbd_study/decision.json). При повторной проверке был исправлен пропущенный поворот RGB в геометрии поз; приведённые ниже числа уже пересчитаны.

Это продолжает [восстановление исходной записи](dobbe_episode_3651_recovery.md). Исходную depth повторно не скачивали.

## Источники и фиксированный временной протокол

| Сцена | HoNY запись | RGB / depth / labels | Использование |
|---|---|---|---|
| A | `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30` | 243 / 243 / 243 | основной красный стакан, связь с ShareRobot `dobbe#episode_3651` и LeRobot `episode_003651` обоснована [ранее](dobbe_episode_3651_recovery.md) |
| B | `Pick_and_Place/Home7/Env1/2023-04-27--10-47-40` | 278 / 278 / 278 | другая запись, другой дом, текстурированный диван и заметный параллакс; [квитанция выборочного извлечения](../runs/dobbe_rgbd_study/second_download_receipt.json) |

RGB имеет `256×256`, depth `192×256 float32` в обеих записях. Во второй сцене из официального split ZIP извлечены **только** `compressed_video_h264.mp4`, `compressed_np_depth_float32.bin`, `labels.json`; архив ~82 ГБ целиком не загружался. Исходные RGB кадры показаны для [сцены A](../runs/dobbe_rgbd_study/main_contact.jpg) и [сцены B](../runs/dobbe_rgbd_study/second_contact.jpg); наложения depth границ приведены ниже. Подробный [аудит размеров и поз](../runs/dobbe_rgbd_study/scene_inspection.json) сохранён.

[protocol.json](../runs/dobbe_rgbd_study/protocol.json) был записан **до** запусков COLMAP и оценки геометрии:

| Сцена | Calibration past | H0,H1,H2 | Future, только после prediction |
|---|---|---|---|
| A | 0–95 | 96,97,98 | 99–128 |
| B | 0–120 | 121,122,123 | 124–153 |

Все вычисления `K`, RGB↔depth, статических точек и pose использовали только calibration past. Сжатый depth поток технически распаковывается целиком, затем немедленно обрезается до разрешённого префикса; будущие значения не участвуют в выборе или расчётах. Разбор полного RGB видео для обзорного contact sheet не применялся к калибровке.

## COLMAP: только RGB

Установлен клон указанного [`f2-nerf`](https://github.com/Devil-SX/f2-nerf/blob/e235005ada08089d0bc16767541f87d45d723a64/scripts/colmap2poses.py), commit `e235005ada08089d0bc16767541f87d45d723a64`. Его `colmap2poses.py` читает `fx,fy,cx,cy` из COLMAP и затем переводит COLMAP poses в NeRF convention; к Dobb·E poses эту NeRF перестановку не применяли. Из официальных Python bindings COLMAP установлен `pycolmap==4.2.1`; его [документация](https://github.com/colmap/colmap/blob/main/python/README.md) описывает те же feature extraction, matching и incremental mapping. Для каждого прогона использованы только PNG RGB кадров префикса без resize/crop, общая камера `CameraMode.SINGLE`, SIFT, exhaustive matching, initial `K=(200,200,128,128)`, затем bundle adjustment. Depth и `labels.json` в базу COLMAP не подавались. Код и настройки: [run_dobbe_rgb_colmap.py](../scripts/run_dobbe_rgb_colmap.py); все параметры, registered image IDs, reprojection и camera centers: [colmap_runs.csv](../runs/dobbe_rgbd_study/colmap_runs.csv) и индивидуальные `summary.json` в `runs/dobbe_rgbd_study/{main,second}/colmap_*`.

### Основная сцена: обязательные 6 прогонов

| Модель | RGB subset | Registered / input | fx, fy (px) | cx, cy (px) | 3D points |
|---|---|---:|---:|---:|---:|
| PINHOLE | all | 9/96 | 1742, 3758 | 305, −165 | 0 |
| PINHOLE | even | 4/48 | 6, 194 | 95, −10 | 127 |
| PINHOLE | odd | 3/48 | 6, 1043 | 99, 118 | 6 |
| OPENCV | all | 2/96 | 514, 457 | 189, 109 | 211 |
| OPENCV | even | 2/48 | 279, 410 | 190, 89 | 113 |
| OPENCV | odd | 4/48 | 220, 241 | 207, −105 | 173 |

OPENCV distortion также нестабилен: например, `k2=97.45` на полном префиксе против `k2=−2.00` на чётном. Малый COLMAP training reprojection error в этих моделях относится лишь к 2–9 зарегистрированным кадрам и не валидирует камеру. Дополнительные прогоны с фиксированной главной точкой и заранее выбранными начальными парами не исправили сцену A: лучший охват 9/96, наиболее удачная фиксированная модель оценила `fx≈2819,fy≈6752` px. [График устойчивости](../runs/dobbe_rgbd_study/colmap_k_stability.png) показывает зарегистрированные кадры и focal lengths.

### Вторая сцена

Свободная главная точка дала только 5/121 кадров и `cy≈−46`. Когда `cx=cy=128` **зафиксированы, а не оценены**, выбранная RGB-only начальная пара 62/107 дала 121/121 кадров, 4911 точек и `fx=232.83,fy=293.92`; средняя ошибка внутри COLMAP `0.49 px`. На нечётном subset восстановлены 60/60 кадров, но `fx=197.50,fy=252.05` (оба примерно на 14–15% ниже). Чётный subset не дал пригодной модели: в лучших автоматических настройках 7/61 кадров, с другой начальной парой 4/61. Оба прогона теперь проверены на полные `1830/1830` RGB-пар в SQLite базе; прерванный вариант с неполной базой из анализа исключён. Это проверка диагностической устойчивости при **предположенном** центре изображения, не независимая оценка полной `K`.

### Дополнительная проверка пропорции RGB

[Код Dobb·E](https://github.com/notmahi/dobb-e/blob/cf06a27e180625754c3508411117c157c615a2a7/stick-data-collection/process_from_r3ds.py) **растягивает** исходные RGB до `256×256`, а измеренная depth экспортируется как `256×192`; поэтому опубликованные квадратные кадры имеют не исходную пропорцию пикселей. Отдельно, после обязательных прогонов на неизменённых кадрах, мы вернули RGB к `256×192` и запустили RGB-only `SIMPLE_PINHOLE` с одним фокусным расстоянием и фиксированным центром `(128,96)`. Это физически мотивированное ограничение, но обратный resize не восстанавливает потерянные детали и результат остаётся диагностическим. `K` ниже переведена обратно к опубликованным `256×256` пикселям с учётом центров пикселей.

| Сцена | RGB subset | Registered / input | `fx,fy` в опубликованных 256×256, px | Вывод |
|---|---|---:|---:|---|
| A | all | 0/96 | — | оптимизация отбросила модель |
| A | even / odd | 3/48 / 3/48 | 1197,1597 / 500,667 | неприемлемая и нестабильная `K` |
| B | all | 121/121 | 233.75,311.66 | полная RGB реконструкция |
| B | even / odd | 6/61 / 60/60 | 12081,16108 / 210.48,280.64 | сильная зависимость от subset |

Значит известная деформация при экспорте **не объясняет** провал калибровки A. Исходные `256×256` прогоны выше остаются основной проверкой требований задачи; этот опыт сохраняется в отдельных `colmap_simple_pinhole_*_rectified` каталогах и [сводной CSV](../runs/dobbe_rgbd_study/colmap_runs.csv).

## RGB↔depth и `K_depth`

Проверена заранее заданная карта координат `u_d=u_r`, `v_d=(v_r+0.5)·192/256−0.5`. На пяти кадрах каждой сцены границы глубины наложены на RGB: [A](../runs/dobbe_rgbd_study/main_depth_edge_overlay.jpg), [B](../runs/dobbe_rgbd_study/second_depth_edge_overlay.jpg). Доля depth-edge пикселей на расстоянии не более 3 px от RGB edge составляет в среднем `0.826` (A) и `0.975` (B), при горизонтальном сдвиге ±8 px — `0.472–0.476` и `0.843–0.863`. [Покадровые числа](../runs/dobbe_rgbd_study/rgb_depth_alignment.json). B содержит густую текстуру, поэтому её edge score менее специфичен; оба наложения нужно оценивать визуально. Пространственная гипотеза **поддержана**, но не доказывает точное субпиксельное соответствие или depth semantics.

При этой карте соответствующий `K_depth` имеет `fx_d=fx`, `fy_d=0.75fy`, `cx_d=cx`, `cy_d=0.75(cy+0.5)−0.5`; коэффициенты искажения в нормализованных координатах сохраняются. Значения для каждой проверенной `K` записаны как `K_depth` в файлах статической геометрии. Для сцены A валидированной `K_depth` нет, поскольку не валидирована сама `K`.

## Независимая метрическая проверка

На calibration past выбраны RGB-соответствия SIFT + ratio test + RGB-only fundamental-matrix RANSAC; в A область ограничена стаканом/столом/неподвижным окружением, в B — верхним фоном дивана вне движущегося рулона. Использована медиана depth `3×3`; точки с разбросом depth >0.08 м исключены. [Визуальная ревизия совпадений A](../runs/dobbe_rgbd_study/main_rgb_matches_0_20.png) и [B](../runs/dobbe_rgbd_study/second_rgb_matches_80_100.png). Получено 94 парных наблюдения A и 72 B, от 9 до 35 на пригодную пару. Это парные наблюдения, **не** 94 уникальные точки, отслеженные через весь ролик; такой более строгий многокадровый тест остался недоступным при слабой калибровке.

Измеренная depth сравнивалась как camera-Z и как длина луча; labels как camera→world и world→camera. Автор Record3D [подтвердил](https://github.com/marek-simonik/record3d/issues/59) OpenGL оси камеры (взгляд вдоль `−Z`), тогда как RGB обратно проецируется в OpenCV оси (`+Z` вперёд, `+Y` вниз). Есть ещё один **обязательный поворот**: [экспортёр Dobb·E](https://github.com/notmahi/dobb-e/blob/cf06a27e180625754c3508411117c157c615a2a7/stick-data-collection/process_from_r3ds.py) поворачивает портретный RGB на 90° по часовой стрелке, а [преобразование labels](https://github.com/notmahi/dobb-e/blob/cf06a27e180625754c3508411117c157c615a2a7/stick-data-collection/utils/action_transforms.py) использует для него `P_old`. Чтобы вернуть луч из сохранённого RGB к исходным осям OpenGL, нужна матрица `R_z(+90°)`. Таким образом `C=P_old·R_z(+90°)·diag(1,−1,−1)`. Прямое умножение показывает `P_old·R_z(+90°)=P_new`; для альбомного экспорта код использует `P_new` без поворота RGB. **Обе ветки экспорта дают один и тот же эффективный `C=P_new·diag(1,−1,−1)` для опубликованного изображения.** Исходную ориентацию записи знать для этой операции не требуется, и произвольные перестановки осей не подбирались. Предыдущий расчёт с `P_old·diag(1,−1,−1)` пропускал поворот RGB; его влияние сохранено в [диагностике](../runs/dobbe_rgbd_study/image_rotation_diagnostic.json), но далее он не используется. `X_i` из RGB, depth и `K` преобразованы через соответствующую pose в общую систему; посчитаны расстояние между наблюдениями одной неподвижной точки (м) и перепроекция A→B (px). Код: [validate_dobbe_geometry.py](../scripts/validate_dobbe_geometry.py); все четыре семантических варианта каждой `K`, P90 и покадровые числа: [geometry_metrics.csv](../runs/dobbe_rgbd_study/geometry_metrics.csv).

Для сопоставимой таблицы ниже зафиксирован вариант `c2w + camera-Z` (labels применяются как camera→world). Это согласуется с [генерацией labels в Dobb·E](https://github.com/notmahi/dobb-e/blob/cf06a27e180625754c3508411117c157c615a2a7/stick-data-collection/process_from_r3ds.py) и даёт меньшую статическую ошибку, чем инверсия labels. Depth semantics остаётся недостаточно определённой.

| Сцена / K | Median 3D, м | P90 3D, м | Median A→B, px | Вывод |
|---|---:|---:|---:|---|
| A naive `(200,200,128,128)` | 0.0767 | 0.1546 | 19.0 | baseline |
| A previous fit `(174,199,154,155)` | 0.0787 | 0.1573 | 13.5 | прежняя подгонка снижает px, но не является независимой RGB оценкой и увеличивает 3D ошибку |
| A COLMAP PINHOLE all | 0.0699 | 0.1478 | 197.8 | медиана 3D лучше, px катастрофически хуже |
| A COLMAP OPENCV all | 0.0714 | 0.1502 | 62.2 | медиана 3D лучше, px существенно хуже |
| B naive | 0.00746 | 0.0729 | 3.66 | baseline |
| B COLMAP PINHOLE all, fixed center | 0.00523 | 0.0732 | 2.07 | медианы лучше, P90 остаётся плохим |
| B COLMAP PINHOLE odd, fixed center | 0.00515 | 0.0742 | 1.61 | медианы лучше, K зависит от subset |
| B aspect-rectified SIMPLE_PINHOLE all | 0.00539 | 0.0727 | 2.23 | медианы лучше, even subset провален |

[График двух независимых метрик](../runs/dobbe_rgbd_study/static_and_reprojection_comparison.png) и примеры перепроекции [A](../runs/dobbe_rgbd_study/main_reprojection_0_20.png), [B](../runs/dobbe_rgbd_study/second_reprojection_80_100.png) фиксируют масштаб ошибок. Для naive K в A прямое `c2w` применение labels даёт `0.0767 м / 19.0 px` против `0.1433 м / 22.7 px` при инверсии labels; в B `0.00746 м / 3.66 px` против `0.0746 м / 32.3 px`. Это вместе с кодом Record3D/Dobb·E поддерживает `c2w`. В B на паре кадров 100→120 даже после исправления остаются ошибки `0.105 м / 39.9 px`; поэтому хороший общий median не следует принимать за гарантию на каждом участке. Z и длина луча различаются в A на ~3.1 мм, в B на ~0.5 мм по медианной 3D ошибке; по перепроекции они также близки. [Спецификация Apple ARDepthData](https://developer.apple.com/documentation/arkit/ardepthdata) определяет LiDAR depth как расстояние от плоскости камеры, то есть camera-Z; [экспортёр Dobb·E](https://github.com/notmahi/dobb-e/blob/cf06a27e180625754c3508411117c157c615a2a7/stick-data-collection/export_vids_ffmpeg.py) лишь декомпрессирует, поворачивает и перепаковывает depth. Это делает camera-Z правдоподобной трактовкой, но исходный `metadata`/тип сенсора данной записи отсутствует, а метрический тест не разделяет варианты достаточно уверенно.

### Устойчивость к качеству RGB-D точек

Отдельная [проверка чувствительности](../runs/dobbe_rgbd_study/correspondence_robustness.json) использовала те же causal RGB-совпадения, удалила близкие дубли (<3 px в любом кадре) и потребовала разброс depth `3×3` ≤2 см в каждом кадре. Это оставило 67 из 94 парных наблюдений A и 56 из 72 B. Правило одинаково для всех `K`; оно **не использовалось для подбора** intrinsics или пересмотра gate. На очищенных точках A медианная перепроекция остаётся `21.5 px` для naive `K`, `225.6 px` для COLMAP PINHOLE и `76.8 px` для OPENCV. Последняя пара A (80→95) влияет на общую медиану: без неё и после фильтра `8.6 px` для naive против `78.3 px` для COLMAP PINHOLE. Следовательно провал основной COLMAP `K` не создаётся дубликатами или одной поздней парой. Код проверки: [stress_dobbe_correspondence_quality.py](../scripts/stress_dobbe_correspondence_quality.py).

## Решение и граница результата

`BLOCKED` обоснован двумя независимыми наблюдениями: RGB-only COLMAP на A не наблюдает устойчивую полную `K`, и ни одна полученная `K` не даёт убедительного **совместного** улучшения статической 3D-согласованности и перепроекции. Известный поворот RGB и исправление осей поз улучшают проверку, но не устраняют эту проблему. B подтверждает, что код может построить COLMAP модель в более текстурированной сцене: там медианы 3D и перепроекции улучшаются совместно. Однако её `K` неустойчива к subset, а P90 ошибки остаются высокими, так что B не даёт калибровку для A. Неизвестную исходную `metadata/K` нельзя заменить цифрами диагностики и назвать результат опубликованной Record3D калибровкой.

Строго валидированный `[3,8,3]` и полные GT/ADE/FDE по восьми точкам не получены. Отдельный [приближённый прогон](dobbe_approx_colmap_f2nerf_molmo.md) с перенесённой через F2-NeRF `K` создал `(3,8,3)`, запустил MolmoMotion и посчитал ошибки на 128 из 240 наблюдений. Продолжение строгой проверки требует источника надёжной `K` **именно опубликованного RGB**, а также объяснения оставшейся статической ошибки A при уже исправленной системе осей. Это может быть исходный `metadata`/R3D конкретной записи или новая, независимо проверенная калибровка, удовлетворяющая тем же метрическим проверкам.

## Воспроизведение

Рабочая среда: WSL Ubuntu, `F:\AIRI_task\.venv`, `pycolmap==4.2.1`, `pyliblzfse==0.4.1`, установленные `numpy`, `scipy`, `opencv-python`, `matplotlib`. Клон Dobb·E: `cf06a27e180625754c3508411117c157c615a2a7`; MolmoMotion: `8091aa98df93495dfa9e27dd998e1932c2806f57`. Для извлечения B использован параметризованный [скрипт выборочной загрузки](../scripts/extract_dobbe_3651_rgbd.py) поверх существующего [индекса split ZIP](../runs/plex_dobbe_preflight/dobbe_depth_nested_index.json).

Из `molmo-motion`:

```bash
python scripts/write_dobbe_protocol.py
python scripts/run_dobbe_rgb_colmap.py --scene main --model PINHOLE --subset all
python scripts/run_dobbe_rgb_colmap.py --scene main --model OPENCV --subset all
# Повторить обе модели с --subset even и --subset odd.
python scripts/run_dobbe_rgb_colmap.py --scene second --model PINHOLE --subset all --fixed-pp --init-pair 62 107
# Отдельная физически мотивированная диагностика пропорции RGB:
python scripts/run_dobbe_rgb_colmap.py --scene main --model SIMPLE_PINHOLE --subset all --fixed-pp --aspect-rectified
python scripts/run_dobbe_rgb_colmap.py --scene second --model SIMPLE_PINHOLE --subset all --fixed-pp --aspect-rectified
# Повторить этот опыт с --subset even и --subset odd.
python scripts/check_dobbe_rgb_depth_alignment.py
python scripts/validate_dobbe_geometry.py --scene main
python scripts/validate_dobbe_geometry.py --scene second
python scripts/probe_dobbe_pose_branch.py
python scripts/stress_dobbe_correspondence_quality.py
python scripts/summarize_dobbe_rgbd_study.py
python scripts/plot_dobbe_rgbd_study.py
python scripts/audit_dobbe_rgbd_study.py
```

Отдельные дополнительные запуски и начальные пары точно перечислены в [colmap_runs.csv](../runs/dobbe_rgbd_study/colmap_runs.csv); повторное выполнение скрипта COLMAP для папки с уже существующим `summary.json` просто печатает сохранённый итог. Для полного пересчёта отдельного прогона следует удалить **только его собственную** папку `runs/dobbe_rgbd_study/{scene}/colmap_{model}_{subset}*`.
