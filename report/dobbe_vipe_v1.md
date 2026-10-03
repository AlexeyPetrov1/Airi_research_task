# Dobb·E: causal ViPE → AllTracker → MolmoMotion

Дата: 2026-10-03. Новый изолированный эксперимент лежит в `runs/dobbe_vipe_v1/`; старые COLMAP результаты сохраняются. На RTX 4070 фактически выполнены ViPE и AllTracker для A/B/C, полный H3-F30 прогноз для C и диагностические прогнозы A с независимой SIFT-проверкой фона. A/B в исходном `no_vda` не прошли заранее записанный геометрический gate; запуск модели для этих исходных входов пропущен. Положительный контроль A имеет ограниченное покрытие позднего prefix/H3 и отдельно описан ниже.

Для эксперимента допускается **ViPE-estimated 3D**, а не только подтверждённый физический GT. Это не снимает необходимость проверить реконструкцию. Ошибки 3D в оценённой системе координат нельзя выдавать за метрический ground truth.

## Протокол

| Сцена | Наблюдаемые raw IDs | H3 | ViPE H3 indices | Future только для оценки |
|---|---|---|---|---|
| A, красная чашка | 0,2,…,98 (50) | 94,96,98 | 47,48,49 | 100,102,…,158 |
| B, рулон скотча | 1,3,…,123 (62) | 119,121,123 | 59,60,61 | 125,127,…,183 |
| C, передняя панель ящика | 0,2,…,60 (31) | 56,58,60 | 28,29,30 | 62,64,…,120 |

Все три исходные записи физически доступны. Выбор `t0=60` для C сделан по наблюдаемому префиксу: ящик открыт, захват приближается к панели; полный горизонт 30×1/15 с помещается в запись. Тексты действий: A — `Pick up the red cup.` (точный прежний текст), B — `Pick up the roll of tape.`, C — `Close the drawer.`. Для B/C это явные инструкции эксперимента, не восстановленные аннотации ShareRobot.

`causal_15hz.mp4` записывается в исходных 256×256 через lossless RGB H.264. Обратное декодирование проверено на полное равенство пикселей, числа и порядка кадров с исходными декодированными кадрами. `frame_map.json` содержит точные raw IDs и SHA256 каждого PNG. Сохранена только измеренная depth разрешённого префикса. LZFSE поток технически распаковывается целиком, затем немедленно отбрасывается будущая часть; она не участвует в вычислениях.

На t0 сохранены ручные маски объектов, исключающие Stick/gripper. В маске семплируются 100 точек K-means с фиксацией seed и привязкой центров к действительным mask pixels. Дополнительно выбираются статические RGB corners вне объекта и захвата. Для C исключено содержимое ящика, потому что оно движется вместе с ним. Сохранены маски, полигоны, queries и визуальные наложения; все они используют только t0 и наблюдаемый префикс.

AllTracker использует авторскую `Net(16)` и опубликованные веса. Поскольку query расположен в последнем наблюдаемом кадре, causal MP4 разворачивается по времени, t0 становится frame 0, затем выход возвращается в исходный порядок. Это избегает forward вызова на одно-кадровом suffix в интеграционном скрипте. Новые точки выбираются по авторским 16-anchor consensus/MAD фильтрам и ray-only smoothing, затем восьми пространственно разнесённым точкам. Абсолютные геометрические проверки считают **несглаженные** точки. В A дополнительно отслеживаются прежние восемь t0 cup queries для отдельного smoke-run.

## Геометрический gate

Эвристические пороги записаны до запуска ViPE в каждом `protocol.json`: median static cross-frame reprojection ≤4 px, p90 ≤10 px, минимум 30 наблюдений из трёх пар; median изменения расстояний жёсткого объекта ≤15% t0 median pair distance; восемь валидных H3 tracks; 16 реальных anchors; относительный размах focal ≤15%; median отношения ViPE/measured depth в [0.5,2.0] отдельно для объекта и фона. Эти пороги допускают прогноз на оценённой геометрии, но не сертифицируют метрический GT.

Важное ограничение: ViPE оценивает shared intrinsics и передаёт одну K всем кадрам. Поэтому постоянные `fx,fy,cx,cy` ожидаются по конструкции и не являются независимым подтверждением устойчивости калибровки. GeoCalib инициализирует `fx=fy`, центр изображения фиксирован при инициализации; квадратный экспорт Dobb·E растягивает исходное изображение. Оценка cross-frame reprojection и depth нужна именно потому, что эта camera model может быть ограниченной для опубликованных кадров.

Статические t0 точки поднимаются в 3D **один раз**, затем через ViPE c2w переносятся в более ранние камеры и сравниваются с независимо отслеживаемыми RGB координатами. Проверка `2D→3D→2D` в одном кадре не используется как доказательство качества. Читаются реальные `inds` ViPE; неполные или несогласованные индексы, неправильные c2w и несопоставимая разрешающая способность depth завершают маршрут ошибкой.

Для модели сохраняются `points_3d_world.npy` (3,8,3), `points_2d_t0.npy` (8,2), `c2w_t0.npy` (4,4). Processor получает world points и c2w ровно один раз. Фактическая equivalence проверка на старых восьми точках A и выбранных точках C подтвердила побитовое совпадение всех tensor inputs, включая токенизированный prompt и anchor, с вручную преобразованной camera-t0 историей. При непройденном gate MolmoMotion пропускается с квитанцией `SKIPPED_GEOMETRY_GATE`.

## Среда и воспроизведение

Windows CUDA Toolkit 13.2 не является Linux-компилятором. В WSL Ubuntu 24.04 установлен Linux CUDA compiler/runtime headers 12.8 и системный C++ compiler. Среда `F:/AIRI_task/.venv-vipe` отдельная от рабочей `.venv` MolmoMotion. Используются vendored ViPE, preset `no_vda`, PyTorch 2.9.1 CUDA 12.8 и Transformers 4.57.1. Первый uv cache на DrvFS прервался из-за rename permission error; новый cache размещён на ext4. При первой CUDA сборке отсутствовали `cusparse.h`; include paths исправлены на CUDA library headers из PyTorch wheels. Системные NVIDIA Linux GPU drivers не устанавливаются: WSL использует драйвер Windows. [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Первый запуск B завершился ошибкой: `gdown 6.4.1` удалил аргумент `fuzzy`, который вызывает vendored ViPE. В отдельной среде закреплён совместимый `gdown==5.2.0`, после чего B выполнена успешно. Первоначальный log и квитанция сохранены отдельно. Сборка CUDA extension заняла около девяти минут; холодный запуск B включает загрузку весов и не является чистым benchmark inference.

Системные зависимости: `wsl -d Ubuntu -u root --exec bash /mnt/f/AIRI_task/molmo-motion/scripts/setup_dobbe_vipe_system.sh`. Python среда: `wsl -d Ubuntu --exec bash /mnt/f/AIRI_task/molmo-motion/scripts/setup_dobbe_vipe_env.sh`. Затем из PowerShell:

```powershell
./scripts/run_dobbe_vipe_v1.ps1 prepare
./scripts/run_dobbe_vipe_v1.ps1 masks
./scripts/run_dobbe_vipe_v1.ps1 vipe B
./scripts/run_dobbe_vipe_v1.ps1 vipe A
./scripts/run_dobbe_vipe_v1.ps1 vipe C
./scripts/run_dobbe_vipe_v1.ps1 track B
./scripts/run_dobbe_vipe_v1.ps1 track A
./scripts/run_dobbe_vipe_v1.ps1 track C
./scripts/run_dobbe_vipe_v1.ps1 geometry B
./scripts/run_dobbe_vipe_v1.ps1 geometry A
./scripts/run_dobbe_vipe_v1.ps1 geometry C
./scripts/run_dobbe_vipe_v1.ps1 variants A
./scripts/run_dobbe_vipe_v1.ps1 processor A smoke_vipe
./scripts/run_dobbe_vipe_v1.ps1 infer A smoke_vipe
./scripts/run_dobbe_vipe_v1.ps1 infer A pure_vipe
./scripts/run_dobbe_vipe_v1.ps1 variants C
./scripts/run_dobbe_vipe_v1.ps1 infer C pure_vipe
```

Следует запускать GPU стадии последовательно; ViPE, AllTracker и MolmoMotion одновременно не загружаются. CPU подготовка и диагностика могут перекрываться с GPU стадией. Для depth-only абляции предусмотрены `pure_vipe_unsmoothed` и `hybrid`: одинаковые точки, текст, ViPE K и c2w, отсутствие smoothing в обеих ветках. Hybrid требует визуально проверенного `hybrid_alignment_review.json` и собственного reprojection/rigidity gate. Перевод ViPE camera translations в другой масштаб по будущему не допускается.

Метод опирается на [официальный pipeline MolmoMotion](https://github.com/allenai/molmo-motion/blob/main/data_generation/README.md) и [ViPE presets](https://nv-tlabs.github.io/vipe/usage/). В отличие от полного grounding pipeline на миллионе видео, здесь объект явно задан сохранённой исследовательской маской.

## Проверки реализации

`tests/test_dobbe_vipe_v1.py`: **9 passed**, один ожидаемый warning о неустановленном `MOLMO_DATA_DIR` (этот адаптер читает локальные файлы напрямую). Проверены точные causal 15-Hz сетки A/B/C, корректная backprojection и reprojection при известном движении камеры, отсутствие clamp для invalid tracks, отклонение missing ViPE indices, преобразование K с учётом центров пикселей при восстановлении пропорции, чувствительность независимого static audit к несовпадению depth/translation scale и совпадение сериализованного prompt и anchor публичного processor для world+c2w и camera-t0 входов. Эти проверки верифицируют адаптер, но не заменяют геометрическую проверку реальных сцен.

## Фактические результаты исходного no_vda

| Сцена | Static median / p90, px | Rigid median, % | Depth ViPE/DobbE: объект / фон | Кандидатов после filter | Gate |
|---|---:|---:|---:|---:|---|
| A | 5.92 / 24.74 | 0.53 | 1.62 / 1.58 | 88 | REJECT: static |
| B | 7.25 / 22.43 | 2.65 | 2.95 / 2.76 | 83 | REJECT: static, depth scale |
| C | 1.35 / 3.07 | 0.72 | 1.19 / 1.22 | 95 | PASS: estimated geometry |

Во всех сценах выбраны восемь точек из первоначальных 100 и использованы 16 реальных anchors. Focal A/B/C в исходных 256×256: 298.85 / 458.89 / 208.82 px. `fx=fy`, principal point `(128,128)`; это результат ограничений shared camera model. Порог gate после результатов не менялся.

В независимом causal SIFT контроле correspondence отбирались без использования ViPE depth/poses: mutual Lowe 0.7 и fundamental RANSAC 1 px. A: median 0.70, p90 10.19 px, 67 observations; надёжных соответствий в ранних парах недостаточно. B: median 5.17, p90 10.48 px, 167 observations. Это указывает на вклад ошибок background tracking и геометрии; исходное решение gate сохранено. Низкая rigidity сама по себе не подтверждает правильность камеры.

| Этап | Время, s | Память |
|---|---:|---|
| B ViPE no_vda, холодный запуск | 392.09 | отдельные RAM/CUDA peaks не записаны |
| A ViPE no_vda | 129.62 | peak child RAM 5.51 GiB |
| C ViPE no_vda | 116.44 | peak child RAM 5.44 GiB |
| AllTracker A / B / C | 2.49 / 3.40 / 2.20 | peak CUDA 1.01 / 1.15 / 0.64 GiB |
| C MolmoMotion, load + predict | 264.95 | peak CUDA 9.64 GiB; peak RAM 22.28 GiB |
| C predict отдельно | 187.26 | BF16, seed 0 |

Время модели начинается после processor и SHA256 checkpoint, поэтому не является временем всего процесса. C выдала конечный массив `(8,30,3)`; все 240 point-frames присутствуют в разобранном тексте, значения конечные. Это подтверждает исполнение модели, а качество прогноза требует отдельной оценки.

Hybrid рассчитан на тех же выбранных queries с сохранением исходных ViPE poses. Наложения measured-depth edges поддерживают грубое RGB/depth соответствие, но у чашки edge верхней границы смещён внутрь: это не pixel-exact calibration. Проверка ограничена наблюдаемыми кадрами и явно описана в `hybrid_alignment_review.json`. A hybrid: median/p90 static 8.38/22.21 px; C hybrid: 5.52/7.84 px. Обе ветки отвергнуты собственным gate. Глубина Dobb·E не исправляет реконструкцию автоматически; её другой масштаб несовместим с неизменёнными ViPE translations. Прогнозный depth-only A/B эксперимент при таких входах не заявляется выполненным.

## Контроли камеры и depth alignment

Дополнительные ветки сохраняют baseline, исходные query IDs, 256×256 PNG модели и пороги. B `default` использует официальный Small Video Depth Anything (`vits`), поместился на 12 GiB и завершился за 176.46 s, peak child RAM 5.54 GiB. Его static median/p90 — 8.31/23.05 px, depth ratios — 3.30/3.28; gate REJECT. Это фактический контроль, а не предполагаемый OOM.

Ветка `rectified` обращает известное растяжение экспортера: causal RGB уменьшается по вертикали 256→192 без апскейла. ViPE использует no_vda на 256×192; depth и K явно переводятся обратно в исходные model pixels. `fy_model = fy_vipe × 256/192`, `cy_model = (cy_vipe+0.5) × 256/192−0.5`. Camera axes и c2w не меняются. AllTracker tracks берутся в исходных пикселях, а не запускаются повторно на другой сетке.

B rectified завершился за 279.99 s, peak child RAM 5.54 GiB. Static median/p90 улучшились до 3.71/12.74 px; depth ratios — 2.30/2.16. Но p90 и scale остаются вне исходного gate, поэтому прогноз также пропущен. K в исходных пикселях: `(206.64,275.51,128.00,128.17)`. Независимый SIFT: median 0.63, p90 30.88 px — медиана улучшилась, хвост ошибок сохранился. Контроль поддерживает влияние camera model, но не доказывает достаточную реконструкцию B.

A rectified завершился за 186.03 s, peak child RAM 5.52 GiB. Static AllTracker median/p90 — 4.53/27.13 px, depth ratios — 1.08/0.94, K — `(145.62,194.16,128.00,128.17)`. Этот основной gate также REJECT. Независимые SIFT matches дали median/p90 0.65/5.93 px на 67 observations в четырёх поздних парах raw 72/92/94/96 относительно t0=98; ранние пары не обеспечили достаточно надёжных correspondences.

### Отдельный exploratory контроль A с независимой проверкой фона

`A/sift_audited` сохраняет те же rectified ViPE outputs, исходные RGB, AllTracker object tracks и выбранные point IDs. Меняется только способ измерения static residuals: фиксированные RGB-only SIFT correspondences вместо ненадёжных background AllTracker tracks. Numeric thresholds остаются прежними. Выбор метода сделан **после** просмотра реконструкционных диагностик, но **до первого прогноза A и доступа к будущему A**; это post-hoc exploratory контроль, а не новый положительный результат исходного gate. Покрытие проверки ограничено поздним prefix/H3, поэтому качество ранней SLAM траектории не установлено.

ViPE static 0.65/5.93 px и measured-depth hybrid static 0.86/4.57 px проходят одинаковые пороги на **тех же** 67 SIFT correspondences. Маски, тексты, K и camera translations не подгоняются. У первоначального hybrid median rigidity 0.59%, но p90 46.3%. Проверка конкретных queries обнаружила sensor Z=1.049 и 0.439 у двух верхних точек при median cup sensor Z≈0.22: RGB/depth correspondence у этих queries не поддерживается. Численные медианы скрывали проблему. Этот hybrid дополнительно отвергнут **до его прогноза**, а arrays сохранены в `rejected_sensor_queries`.

Первыми выполнены legacy-eight smoke, выбранные восемь со smoothing и те же восемь без smoothing. Все три дали 240 из 240 parsed point-frames. Старые eight queries сохранены для качественного сопоставления, но старый COLMAP прогноз имел другую временную сетку, поэтому это не чистая geometry-only абляция с прошлым результатом.

Для depth-only сравнения создан отдельный общий пул **до первого paired прогноза и доступа к будущему A**: пересечение авторского ViPE filter с valid sensor H3, patch spread <10% sensor Z и sensor Z в [0.5,1.5] от median object-query Z каждого H3 кадра. Это явно записанный post-hoc контроль по наблюдаемым входам, а не подгонка по future/prediction error. Из 72 совместно поддержанных кандидатов выбраны IDs `[59,74,13,19,71,81,8,35]`; один и тот же набор используется в `paired_vipe` и `paired_hybrid`, обе истории несглажены. Различие историй: median 0.0314, p90 0.0596 оценённых единиц; sensor Z t0 выбранных точек 0.203–0.281. Median/p90 hybrid rigidity — 0.41%/28.47%: остаточная неоднородность сенсора и приблизительная registration сохраняются.

Веса модели можно загрузить один раз, seed=0 устанавливается для каждого варианта; RAM peak при этом относится ко всему процессу. Исходные решения A/no_vda и A/rectified остаются REJECT. Между исходным `pure_vipe` и paired-вариантами отличаются point IDs; строгое сравнение источников depth проводится только внутри paired-пары.

### A: полученные прогнозы и условная оценка

Все пять разрешённых вариантов завершились успешно: `(8,30,3)`, 240/240 parsed point-frames, processor equivalence PASS. Predict times для smoke / pure / unsmoothed / paired ViPE / paired hybrid — 152.82 / 139.61 / 145.12 / 144.98 / 136.74 s. Максимальный CUDA allocated peak — 9.615 GiB, process RAM peak — 22.40 GiB. Первоначальный sensor-unsafe hybrid имеет `SKIPPED_GEOMETRY_GATE`; его прогноз не запускался.

Smoke и оба исходных ViPE-варианта, а также paired ViPE предсказывают постоянные 3D координаты на всём горизонте: различие с t0 находится на уровне 0.00035–0.00245 оценённых единиц и включает квантование координат. Paired hybrid добавляет небольшую динамику одной точки (maximum temporal coordinate change 0.011). Замена depth сама по себе не приводит к прогнозу выраженного движения чашки.

Будущее A открыто после завершения **всех пяти** вариантов; квитанция каждой оценки содержит hashes всех пяти уже сохранённых predictions. На RGB чашка переносится с захватом/камерой, а фон заметно меняется. Почти постоянные image coordinates чашки не означают неподвижность объекта в world frame.

| Вариант A | Conditional pixel ADE / FDE | Stationary ADE | Сравниваемых point-frames |
|---|---:|---:|---:|
| legacy-eight smoke | 218.01 / 469.29 | 218.41 | 240/240 |
| selected eight, smoothed | 222.11 / 483.22 | 222.55 | 240/240 |
| те же eight, unsmoothed | 221.94 / 482.60 | 222.59 | 240/240 |
| common paired eight, ViPE | 218.83 / 468.29 | 219.23 | 240/240 |
| те же common eight, measured depth | 225.65 / 485.45 | 225.62 | 240/240 |

Эти значения **не являются достоверной ошибкой модели**: published-pose hypothesis не прошла даже observed-prefix audit A (median 35.71, p90 100.96 px), а источники K/depth остаются оценёнными. В этом условном сравнении stationary baseline почти совпадает с моделью. Разницу ADE paired 218.83→225.65 нельзя использовать как доказательство превосходства ViPE depth над сенсором. Подтверждённый результат абляции — изменение входной 3D-истории и слабая чувствительность predicted motion; метрическое качество переноса не установлено.

## C: оценка после фиксации прогноза

Будущие raw IDs 62,64,…,120 впервые декодированы для этого маршрута после успешного прогноза и проверки SHA256 его входов. `future_access_receipt.json` фиксирует prediction hash **до** доступа к будущему. Будущие 2D correspondences получены отдельным forward AllTracker от тех же t0 queries; модельные входы и gate не менялись.

Исходная K и калиброванные camera poses Dobb·E отсутствуют. Поэтому сравнение проекции с будущими RGB tracks является **условной диагностикой**, использующей published labels pose/basis hypothesis и causal ViPE K. Эта гипотеза сама не прошла observed-prefix static audit: median 8.57, p90 14.28 px. Будущая tracking visibility также оценена моделью. Результат нельзя подавать как проверенную 3D ADE/FDE или сертифицированную 2D ошибку MolmoMotion.

На 152 из 240 point-frames (63.3%) условная mean pixel ADE — 488.85 px, FDE по четырём видимым точкам последнего кадра — 1267.45 px; stationary t0-3D baseline при той же pose hypothesis — mean ADE 168.26 px. Большие значения намеренно не обрезаются границами изображения. На видео ящик закрывается, тогда как forecast расходится с observed tracks уже на раннем горизонте и затем покидает изображение. Прогноз предсказывает примерно 0.55–0.57 оценённых единиц перемещения к двум секундам. **Успешный запуск не означает успешный перенос качества модели на эту сцену.** Ошибки predicted motion и гипотезы evaluation poses здесь не разделены строгим GT.

Для повторения контролей и оценки:

```powershell
./scripts/run_dobbe_vipe_v1.ps1 -Stage vipe -Scene B -Branch default
./scripts/run_dobbe_vipe_v1.ps1 -Stage geometry -Scene B -Branch default
./scripts/run_dobbe_vipe_v1.ps1 -Stage vipe -Scene B -Branch rectified
./scripts/run_dobbe_vipe_v1.ps1 -Stage geometry -Scene B -Branch rectified
./scripts/run_dobbe_vipe_v1.ps1 -Stage variants -Scene C
./scripts/run_dobbe_vipe_v1.ps1 -Stage infer -Scene C
wsl -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/evaluate_dobbe_vipe.py --scene C
```

Отдельный A контроль (после rectified geometry и `audit_dobbe_vipe_static.py --scene A --branch rectified`):

```powershell
wsl -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/prepare_dobbe_vipe_sift_control.py
wsl -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/run_dobbe_vipe_sift_inference.py
wsl -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/evaluate_dobbe_vipe.py --scene A --branch sift_audited --variant paired_vipe
wsl -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/evaluate_dobbe_vipe.py --scene A --branch sift_audited --variant paired_hybrid
```

Повторный запуск inference проверяет hashes cached prediction и всех frozen inputs. Для нового протокола следует создавать новую именованную ветку, сохраняя прежние квитанции.

## Комплект для проверки результата

[Компактные артефакты](dobbe_vipe_results/manifest.json) содержат протоколы, H3 RGB, маски/queries, ViPE K/poses, geometry gates, input arrays, actual predictions, raw output, resource receipts, conditional future comparisons, изображения и MP4. Manifest фиксирует SHA256 и исходный путь каждого файла, а также hashes кода и использованных авторских модулей. Это локальный комплект; внешняя публикация не выполнялась. Полные RGB-D, per-frame ViPE depth и веса исключены из комплекта и остаются в `runs/`/`data/`.

[A: depth-only comparison](dobbe_vipe_results/A/sift_audited/paired_depth_comparison.png), [A: future overlay](dobbe_vipe_results/A/sift_audited/paired_hybrid/evaluation/future_overlay.mp4), [C: forecast](dobbe_vipe_results/C/pure_vipe/forecast_3d.png), [C: future overlay](dobbe_vipe_results/C/pure_vipe/evaluation/future_overlay.mp4). [Verification receipt](dobbe_vipe_results/verification.json) подтверждает causal frame maps, unchanged frozen input hashes, шесть полных прогнозов и одинаковые paired RGB/2D queries/c2w. Исходные A/B gates остаются отрицательными; отдельный A SIFT контроль не заменяет полноценную калибровку.

Полные локальные данные: [A diagnostics](../runs/dobbe_vipe_v1/A/geometry_diagnostics.png), [B diagnostics](../runs/dobbe_vipe_v1/B/geometry_diagnostics.png), [C diagnostics](../runs/dobbe_vipe_v1/C/geometry_diagnostics.png), [C forecast](../runs/dobbe_vipe_v1/C/pure_vipe/forecast_3d.png), [future overlay](../runs/dobbe_vipe_v1/C/pure_vipe/evaluation/future_comparison.png), [conditional metrics](../runs/dobbe_vipe_v1/C/pure_vipe/evaluation/metrics.json). `vipe/`, `tracks.npz`, протоколы, model inputs, квитанции и logs находятся рядом. Для полного повторения нужны локальные исходные RGB-D и checkpoint, а не только отчёт.
