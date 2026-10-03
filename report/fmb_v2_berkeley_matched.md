# FMB v2 — Berkeley-matched pipeline

На основном episode_5201 большая ошибка H3 сохраняется: ADE 134.63 px против 38.22 у static и 69.25 у constant velocity. FDE H3 108.77 px хуже static (69.34), но лучше constant velocity (134.88). На втором FMB-контроле H3 почти совпадает со static и уступает constant velocity по основным 2D-метрикам. Следовательно, тезис о существенном проигрыше обоим baseline по обеим метрикам на обеих сценах не подтверждён. Это независимый повтор на зафиксированных окнах, с новым MolmoPoint → SAM 2.1 → 100 K-means-кандидатов → AllTracker → 24 точки, observed-only UniDepthV2 K и sensor Z16. Результат не подбирался по будущим ошибкам. [Проверка завершения](../runs/fmb_v2_berkeley_matched/completion_audit.json) подтверждает полные прогнозы, формы, общие маски, видео и неизменность 914 файлов FMB v1.

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
| episode_5201 | MolmoMotion_H3 | 134.625 | 108.767 | 0.1308 | 0.1406 |
| episode_5201 | static | 38.222 | 69.342 | 0.0352 | 0.0663 |
| episode_5201 | constant_velocity | 69.246 | 134.881 | 0.0691 | 0.1308 |
| episode_5201 | MolmoMotion_H1 | 125.794 | 208.984 | 0.1378 | 0.2500 |
| fmb_control_n3 | MolmoMotion_H3 | 28.778 | 46.255 | 0.0295 | 0.0470 |
| fmb_control_n3 | static | 28.747 | 46.217 | 0.0300 | 0.0475 |
| fmb_control_n3 | constant_velocity | 22.341 | 33.588 | 0.0409 | 0.0702 |

Помимо ADE/FDE сохранены error(t), per-point ADE, fraction outside image, endpoint displacement, direction error и amplitude ratio в `evaluation/metrics.json` каждой сцены.

H1 на основном эпизоде даёт ADE 125.79 px и FDE 208.98 px: средняя ошибка немного меньше H3, конечная существенно выше, и преимущество static сохраняется. Удаление истории при этом диагностическом checkpoint не устранило большую ошибку. У H3 47.08% оцениваемых позиций находятся вне кадра, у H1 — 20.63%; все они сохранены в метриках.

В n3 прогноз H3 почти неподвижен: среднее смещение на конце около 0.55 px при reference около 46 px. По secondary 3D_est порядок методов отличается: на основном эпизоде H3 хуже обоих baseline по ADE/FDE, а в n3 H3/static имеют меньшую 3D_est ошибку, чем constant velocity, хотя constant velocity лучше в 2D. Различие reference-масок, глубины и оценённой K не позволяет отождествлять эти две метрики или выбирать primary после просмотра результатов.

Чистое время генерации трёх групп и максимальная выделенная CUDA-память:

| Сцена | Checkpoint | Генерация, s | Peak allocated, GiB |
| --- | --- | ---: | ---: |
| episode_5201 | MolmoMotion-4B-H3-F30 | 414.74 | 9.63 |
| episode_5201 | MolmoMotion-4B-H1-F32 | 462.95 | 9.64 |
| fmb_control_n3 | MolmoMotion-4B-H3-F30 | 372.79 | 9.57 |

Загрузка H3 выполнялась один раз для двух сцен, H1 — отдельной загрузкой. В таблице не учитываются чтение весов, подготовка, скачивание и ожидание. Reserved CUDA и RAM сохранены в group receipts.

На основном эпизоде среднее расхождение ECC и AllTracker для **тех же** 24 IDs — 2.038 px, p90 — 4.633 px. Отношение среднего расхождения к H3 ADE — 0.015. [Полная диагностика](../runs/fmb_v2_berkeley_matched/episode_5201/evaluation/reference_uncertainty.json). Визуальная выборка будущих кадров проверена отдельно; оба трекера могут совместно ошибаться на почти безтекстурной грани. Нельзя интерпретировать это отношение как доверительный интервал истинного GT.

Для контрольного n3 дополнительно выполнен тот же ECC-аудит: среднее расхождение 4.457 px, p90 7.734 px. Разницу H3 и static около 0.03–0.04 px нельзя трактовать как содержательное преимущество одного метода. В поздних кадрах нижние/верхние точки могут приближаться к границе детали и захвату; их точная материальная идентичность не доказана. [Визуальный аудит](../runs/fmb_v2_berkeley_matched/visual_review.json).

## Визуальные результаты

Видео показывает первую группу из восьми постоянных IDs для читаемости; метрики и графики полного размаха используют все 24. Зелёный — AllTracker reference, розовый — H3. MP4 содержат 20 реальных future кадров при номинальных 10 FPS.

### episode_5201

K_eff: fx=270.504, fy=268.410, cx=127.818, cy=127.662 px. Coverage: 480/480 в 2D, 436/480 в 3D_est. Максимальная поправка H3 ray smoothing: 6.01 mm. [Метрики](../runs/fmb_v2_berkeley_matched/episode_5201/evaluation/metrics.json), [input audit](../runs/fmb_v2_berkeley_matched/episode_5201/geometry/input_audit.json).

**8 observed RGB**

![8 observed RGB](../runs/fmb_v2_berkeley_matched/episode_5201/viz/observed_8_frames.png)

**t0 + MolmoPoint**

![t0 + MolmoPoint](../runs/fmb_v2_berkeley_matched/episode_5201/observed/molmopoint_overlay.png)

**SAM mask**

![SAM mask](../runs/fmb_v2_berkeley_matched/episode_5201/observed/mask_overlay.png)

**100 candidates**

![100 candidates](../runs/fmb_v2_berkeley_matched/episode_5201/viz/mask_and_100_queries.png)

**24 fixed points**

![24 fixed points](../runs/fmb_v2_berkeley_matched/episode_5201/viz/selected_24_points.png)

**Observed AllTracker tracks**

![Observed AllTracker tracks](../runs/fmb_v2_berkeley_matched/episode_5201/viz/historical_tracks.png)

**Sensor depth**

![Sensor depth](../runs/fmb_v2_berkeley_matched/episode_5201/viz/sensor_depth.png)

**Observed K**

![Observed K](../runs/fmb_v2_berkeley_matched/episode_5201/viz/intrinsics_stability.png)

**RGB/depth registration audit**

![RGB/depth registration audit](../runs/fmb_v2_berkeley_matched/episode_5201/viz/rgb_depth_registration_audit.png)

**Fixed-camera check**

![Fixed-camera check](../runs/fmb_v2_berkeley_matched/episode_5201/viz/fixed_camera_check.png)

**Historical XYZ clouds**

![Historical XYZ clouds](../runs/fmb_v2_berkeley_matched/episode_5201/viz/history_xyz_clouds.png)

**Real future + AllTracker**

![Real future + AllTracker](../runs/fmb_v2_berkeley_matched/episode_5201/viz/real_future_alltracker.png)

**MolmoMotion vs real**

![MolmoMotion vs real](../runs/fmb_v2_berkeley_matched/episode_5201/viz/prediction_vs_real_contact.png)

**All methods, full extents**

![All methods, full extents](../runs/fmb_v2_berkeley_matched/episode_5201/viz/full_extent_trajectories.png)

**Error vs time**

![Error vs time](../runs/fmb_v2_berkeley_matched/episode_5201/viz/error_vs_time.png)

**Per-point ADE**

![Per-point ADE](../runs/fmb_v2_berkeley_matched/episode_5201/viz/per_point_ADE.png)

[Видео рядом](../runs/fmb_v2_berkeley_matched/episode_5201/viz/side_by_side.mp4) · [Видео наложения](../runs/fmb_v2_berkeley_matched/episode_5201/viz/prediction_vs_real.mp4)

**H1 vs H3**

![H1 vs H3](../runs/fmb_v2_berkeley_matched/episode_5201/viz/h1_vs_h3.png)

**Reference uncertainty**

![Reference uncertainty](../runs/fmb_v2_berkeley_matched/episode_5201/viz/reference_uncertainty.png)

**Visual reference audit**

![Visual reference audit](../runs/fmb_v2_berkeley_matched/episode_5201/viz/manual_reference_audit.png)

### fmb_control_n3

K_eff: fx=291.562, fy=291.647, cx=128.037, cy=128.069 px. Coverage: 480/480 в 2D, 423/480 в 3D_est. Максимальная поправка H3 ray smoothing: 17.45 mm. [Метрики](../runs/fmb_v2_berkeley_matched/fmb_control_n3/evaluation/metrics.json), [input audit](../runs/fmb_v2_berkeley_matched/fmb_control_n3/geometry/input_audit.json).

**8 observed RGB**

![8 observed RGB](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/observed_8_frames.png)

**t0 + MolmoPoint**

![t0 + MolmoPoint](../runs/fmb_v2_berkeley_matched/fmb_control_n3/observed/molmopoint_overlay.png)

**SAM mask**

![SAM mask](../runs/fmb_v2_berkeley_matched/fmb_control_n3/observed/mask_overlay.png)

**100 candidates**

![100 candidates](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/mask_and_100_queries.png)

**24 fixed points**

![24 fixed points](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/selected_24_points.png)

**Observed AllTracker tracks**

![Observed AllTracker tracks](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/historical_tracks.png)

**Sensor depth**

![Sensor depth](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/sensor_depth.png)

**Observed K**

![Observed K](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/intrinsics_stability.png)

**RGB/depth registration audit**

![RGB/depth registration audit](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/rgb_depth_registration_audit.png)

**Fixed-camera check**

![Fixed-camera check](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/fixed_camera_check.png)

**Historical XYZ clouds**

![Historical XYZ clouds](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/history_xyz_clouds.png)

**Real future + AllTracker**

![Real future + AllTracker](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/real_future_alltracker.png)

**MolmoMotion vs real**

![MolmoMotion vs real](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/prediction_vs_real_contact.png)

**All methods, full extents**

![All methods, full extents](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/full_extent_trajectories.png)

**Error vs time**

![Error vs time](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/error_vs_time.png)

**Per-point ADE**

![Per-point ADE](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/per_point_ADE.png)

[Видео рядом](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/side_by_side.mp4) · [Видео наложения](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/prediction_vs_real.mp4)

**Reference uncertainty**

![Reference uncertainty](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/reference_uncertainty.png)

**Visual reference audit**

![Visual reference audit](../runs/fmb_v2_berkeley_matched/fmb_control_n3/viz/manual_reference_audit.png)

## Воспроизведение и интерпретация

[Команды](../runs/fmb_v2_berkeley_matched/REPRODUCE.md), [код export](../scripts/fmb_v2_scene.py), [preprocess](../scripts/fmb_v2_preprocess.py), [inference](../scripts/fmb_v2_infer.py), [evaluation](../scripts/fmb_v2_evaluate.py), [проверки протокола](../tests/test_fmb_v2_protocol.py). Checkpoint revisions и хэши сохранены в receipts; сырьё и веса требуется получить отдельно. Все новые файлы находятся в `runs/fmb_v2_berkeley_matched`; скрипты отказываются перезаписывать зафиксированные входы.

Ответ относится к этим двум окнам и конкретной оценённой геометрии. Унификация убирает прежние ручные восемь точек, K_nominal и ECC как основной reference; она не превращает UniDepth K, шкалу depth или безтекстурные соответствия в точную калибровку. В частности, улучшение/ухудшение v1→v2 нельзя приписать единственной причине: одновременно меняются K, выбор и группировка точек, RGB-подготовка и tracker reference. Обобщение на FMB/робототехнику в целом по двум эпизодам не обосновано.
