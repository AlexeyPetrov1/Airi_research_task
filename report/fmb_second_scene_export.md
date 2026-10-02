# FMB: полная независимая вторая сцена

| Field | episode_5201 | second scene |
|---|---|---|
| source `.npy` | `1_M_L_3_vertical_n_2.npy` | `1_L_L_4_vertical_n_0.npy` |
| `trajectory_id` | `2` | `0` |
| N | 148 | 142 |
| shape | 1, rectangle | 1, rectangle |
| size | M, medium | L, large |
| length | L, long | L, long |
| color | 3, Jeans Red | 4, yellow |
| angle | vertical | vertical |
| distractor | n | n |
| primitives | grasp → move_up → go_to_board → insert | grasp → move_up → go_to_board → insert |
| ShareRobot mapping | episode_5201: 2 кадра сопоставлены с raw | ID и 30 кадров не установлены |
| RGB | 4 потока в raw | 4 потока × 142 RGB PNG и 4 MP4 |
| Depth | 4 потока в raw | 4 потока × 142 исходных `uint16` |
| Calibration | active 256×256 K не подтверждена | опубликованные профили четырёх камер; active K не подтверждена |
| Sensor RGB-D readiness | PARTIAL | PARTIAL |
| CAD/PnP readiness | PARTIAL | PARTIAL |
| Stereo readiness | PARTIAL | PARTIAL |
| TCP check readiness | READY | READY |
| Monocular depth readiness | READY | READY |
| 2D evaluation readiness | READY | READY |

## Независимость и выбор

[Официальное правило имён FMB](https://functional-manipulation-benchmark.github.io/dataset/index.html) определяет последний токен имени как `trajectory_id`; оба ID выше получены из имён, а не приравнены к ShareRobot episode. Исходный [5201](../runs/fmb_second_scene/reference_5201.json) и [выбранный run](../runs/fmb_second_scene/selection.json) имеют разные исходные файлы, ID и SHA-256. Новый run скачан как [один оригинальный `.npy` из HF-зеркала FMB](https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/blob/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset/1_L_L_4_vertical_n_0.npy), без архива на сотни ГБ. Его размер 186 224 561 байт; SHA-256 `e6c3aa8e518d75b003a78ac405e33aa30c2ddc94dea2901a5ee568e0fb353cdb`. Копия [source_demo.npy](https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/blob/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset/1_L_L_4_vertical_n_0.npy) побитово совпадает. Исходный официальный архив указан в [metadata.json](../data/fmb_second_scene/metadata.json).

Сначала построена [таблица из 10 кандидатов](../runs/fmb_second_scene/candidate_selection.csv), включая исходный 5201 и отвергнутые варианты. Выбранный объект отличается по **size и color**, но сохраняет тот же тип манипуляции. [Кадры side_1](../runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_1.png) показывают широкую жёлтую грань на всём интересующем движении; [side_2](../runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_2.png) полезна в начале, а позже предмет у верхнего края изображения. Конкурирующий круглый тёмно-синий объект имеет другую shape, но на поздних кадрах слабее отделяется от синей доски и захвата. Модель и результаты её прогноза в выборе не использовались. Равного лидера не было, поэтому seed 42 не потребовался.

Жёлтая внутренняя область проверена цветовой маской **только как диагностикой**: медианная доля положительной raw depth среди кадров с площадью маски ≥30 пикселей равна 0.933 на side_1 и 0.981 на side_2. Маска, отдельные кадры и покадровые доли находятся в [candidate_depth_roi.json](../runs/fmb_second_scene/candidate_depth_roi.json). Это свидетельствует о доступности depth внутри видимой детали, но не доказывает точное совмещение RGB и depth по каждому пикселю.

## Полный экспорт

В [data/fmb_second_scene](../data/fmb_second_scene/metadata.json) сохранены все 142 шага. Исходный FMB хранит камеры в BGR согласно [описанию формата](https://functional-manipulation-benchmark.github.io/dataset/index.html). PNG в `rgb/{side_1,side_2,wrist_1,wrist_2}/` имеют RGB и получены одним обращением порядка каналов; исходный `.npy` не менялся. Четыре [видео](../runs/fmb_second_scene/previews/preview_side_1.mp4) содержат frame index, primitive, gripper state и TCP XYZ. Остальные MP4 находятся в том же каталоге.

Все значения `obs/{side_1,side_2,wrist_1,wrist_2}_depth` сохранены в `data/fmb_second_scene/depth_raw/` как отдельные исходные массивы `uint16` формы `142×256×256`. [depth_statistics.json](../runs/fmb_second_scene/depth_statistics.json) содержит dtype, shape, диапазон, медиану, процентили, доли нулей и nonfinite, число уникальных значений и покадровую долю положительных отсчётов для **каждой** камеры. Side_1: 0…8784, медиана 3483, нули 5.64%; side_2: 0…10301, медиана 4022, нули 3.23%. Wrist камеры содержат значение 65535; его физический смысл здесь не установлен. Цветные depth изображения в `visuals/` созданы только для просмотра. Единица depth и `depth_scale` остаются **UNRESOLVED**.

[robot_state.npz](../data/fmb_second_scene/robot_state.npz) содержит все `tcp_pose`, `tcp_vel`, `tcp_force`, `tcp_torque`, `q`, `dq`, `jacobian`, `gripper_pose`, `action`, `primitive` и JSON представление `object_info`. Исходное поле называется `actions`; оно сохранено в NPZ под ключом `action`, а точное соответствие ключей указано в metadata. В исходном файле нет `object_id`. Диапазоны primitive: grasp 0–54, move_up 55–60, go_to_board 61–84, insert 85–141.

## Калибровка и объект

Неизменённые [четыре официальных файла intrinsics](../data/fmb_second_scene/calibration/raw/side_1) и копии исходного кода камеры и среды находятся в `calibration/raw/`; их URL и SHA-256 перечислены в [asset_provenance.json](../data/fmb_second_scene/asset_provenance.json). [calibration.json](../data/fmb_second_scene/calibration/calibration.json) содержит все опубликованные rectified profiles, включая 640×480 `rectified.2`, числовые параметры исходных JSON с provenance каждого поля, серийные номера из кода и список неизвестных величин. [Исходный код захвата](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/camera/rs_capture.py) запрашивает BGR8, Z16 и выравнивание depth к color при 640×480. [Код среды](https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/envs/franka_fmb_env.py) явно меняет размер RGB до 256×256, но не объясняет, как опубликованная depth стала 256×256. Raw демонстрация не содержит серийного номера камеры. Поэтому active RGB/depth K для опубликованного 256×256, distortion, точная RGB↔depth регистрация, side_1↔side_2, camera↔robot-base и wrist extrinsics не объявлены подтверждёнными. K из episode_5201 не переносилась.

[Официальный STEP со всеми 54 peg телами](../data/fmb_second_scene/object/peg_official.step) и [лист имён формы/цвета](../data/fmb_second_scene/object/shape_color_reference.pdf) сохранены вместе с [object_metadata.json](../data/fmb_second_scene/object/object_metadata.json). В CAD есть геометрия семейства, но тела не подписаны per-run ID. [Диагностический кандидат тела](../data/fmb_second_scene/object/candidate_cad_body.json) с индексом 52 и габаритами примерно 49.81×32.02×150 мм выведен из масштабного семейства официального STEP; его связь с этой демонстрацией **не доказана**, так что это не ground truth объекта и не готовые 2D↔3D соответствия.

## ShareRobot и готовность методов

[sharerobot_mapping.json](../runs/fmb_second_scene/sharerobot_mapping.json) документирует поиск и отдельное препятствие. Из закреплённого ShareRobot `trajectory.json` получены **все 48 индивидуально доступных FMB Trajectory RGB**. [Прямой пиксельный поиск](../runs/fmb_second_scene/sharerobot_trajectory_search.json) сравнил каждый с каждым из 142 шагов всех четырёх исходных камер в двух трактовках каналов после уменьшения до 32×32. У лучшей по этому критерию пары full-resolution MAE равна 18.45, и [сама пара](../runs/fmb_second_scene/visuals/sharerobot_search_best_pair.png) визуально показывает другой синий предмет против выбранного жёлтого. Следовательно, совпадение в этом подмножестве не подтверждено; полный Planning-архив этим поиском не охвачен. Для нового исходного `.npy` не найден source-backed ShareRobot episode ID и не получены его 30 sampled RGB. Соответствие двух кадров старого `episode_5201` нельзя переносить на другой run; `trajectory_id=0` нельзя считать ShareRobot episode ID. Соответственно, grid подтверждённых пар ShareRobot↔FMB не изготовлен и никакое `linspace` сопоставление не выдумано. Это не мешает использовать полный исходный FMB run.

[method_readiness.json](../runs/fmb_second_scene/method_readiness.json) отделяет данные от недоказанных параметров. **MoGe-2, MoGe-3, UniDepthV2, Metric3Dv2 и Depth Anything V2 Metric** имеют полный исходный RGB для будущего запуска; ни одна модель здесь не запускалась. Робототехническую согласованность состояний и независимую будущую **2D** разметку можно выполнять по синхронному run. Для метрического sensor RGB-D пока требуются подтверждённые depth scale, active K и соответствие RGB↔depth. Для CAD/PnP нужны точный body ID, видимые CAD↔RGB опорные точки и active K. Для стерео нужны active K двух сторон и их взаимная поза.

## Визуальный preflight, t0 и проверка

В `visuals/` находятся [contact sheet side_1](../runs/fmb_second_scene/visuals/contact_side_1.png), [side_2](../runs/fmb_second_scene/visuals/contact_side_2.png), [синхронные четыре камеры](../runs/fmb_second_scene/visuals/synchronized_four_cameras.png), [RGB/depth side_1](../runs/fmb_second_scene/visuals/rgb_depth_side_1.png) и [side_2](../runs/fmb_second_scene/visuals/rgb_depth_side_2.png), [доля валидной depth](../runs/fmb_second_scene/visuals/valid_depth_fraction.png), [TCP XYZ](../runs/fmb_second_scene/visuals/tcp_xyz.png), [изменение ориентации](../runs/fmb_second_scene/visuals/tcp_orientation_change.png), [primitive](../runs/fmb_second_scene/visuals/primitive_timeline.png), [gripper](../runs/fmb_second_scene/visuals/gripper_timeline.png) и [конфигурация объекта](../runs/fmb_second_scene/visuals/object_configuration.png). Единицы подписаны только для подтверждённых величин.

[t0_candidates.json](../runs/fmb_second_scene/t0_candidates.json) содержит пять потенциальных t0: 94, 98, 102, 106, 110. Для каждого записаны primitive, число доступных исторических и будущих кадров, маска видимости, смещение центроида детали и TCP, depth, окклюзия и причина интереса. Все пять допускают H=3/F=30; окончательный t0 здесь **не выбирается**. `observed/` и `evaluation/` содержат правила последующего разделения: кадры позже замороженного t0 не должны участвовать в подготовке входной геометрии и масштаба.

[Независимый аудит экспорта](../runs/fmb_second_scene/data_audit.json) прошёл **14/14** проверок: SHA-256, иной исходный run/ID, одинаковое N, все PNG пиксельно равны одному BGR→RGB преобразованию, все depth и state массивы равны raw, MP4 декодируются до 142 кадров каждый, TCP/action конечны, primitive интервалы полны, формы и dtype ожидаемы, RGB/depth меняются во времени. Этот аудит доказывает верность выгрузки, но не разрешает неопределённую метрическую калибровку.

## WHAT IS NOW POSSIBLE

- Запускать позже MoGe-2, MoGe-3, UniDepthV2, Metric3Dv2 и Depth Anything V2 Metric на полном RGB выбранной сцены.
- Проверять per-frame TCP/gripper/action/primitive, траекторию движения и согласованность синхронных данных.
- Размечать и отслеживать видимые точки жёсткой жёлтой детали на полном будущем RGB и строить независимую 2D оценку после заморозки t0.
- Выполнять диагностическую работу с raw depth и официальным CAD семейством, сохраняя неопределённость scale, active K и body ID. Метрический sensor RGB-D, точный CAD/PnP и калиброванное стерео пока не готовы.

PARTIAL
