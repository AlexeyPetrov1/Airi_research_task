# MolmoMotion: вход, прогноз и наблюдаемое движение

Подготовлено 2 октября 2026 года. Два примера имеют разные научные сообщения. Каждый MP4 начинается с трёх настоящих входных кадров и всех восьми выбранных точек с ID 0–7, затем показывает прогноз на реальном будущем RGB. В обоих PNG входные кадры также включены сверху. Продолжительность каждого видео — 8,3 с: вход 2 с, движение при скорости воспроизведения 0,5×, короткие паузы.

| Пример | Видео | PNG для отчёта |
|---|---|---|
| FMB: прогноз и реальное движение | [MP4](fmb_prediction_vs_reality.mp4) | [PNG](fmb_prediction_vs_reality_summary.png) |
| Dobb·E: влияние двух оценок K | [MP4](dobbe_two_colmap_molmo_comparison.mp4) | [PNG](dobbe_two_colmap_molmo_comparison_summary.png) |

## FMB / episode_5201

Запись `1_M_L_3_vertical_n_2.npy`, неподвижная `side_1`. Вход: кадры 124, 125, 126; будущее: 127–146. Зелёный — существующий image-derived RGB proxy GT, оранжевый — настоящий sensor-depth прогноз MolmoMotion, фиолетовый — существующая constant-velocity baseline. Stationary в основной рисунок не добавлена. Все 160/160 будущих RGB-наблюдений доступны.

Слева — реальный RGB с текущими восемью точками и постепенно открываемыми траекториями. Справа — полная плоскость проекции: прогнозы за границами изображения сохраняются. Линии показывают среднее восьми точек; кружки — индивидуальные позиции. В PNG выбраны настоящие кадры 126, 133, 139, 146, то есть 0, 0,70, 1,30, 2,00 с. Они ближайшие к запрошенным 0,67 и 1,33 с.

На этом зафиксированном sensor-входе ADE/FDE MolmoMotion — 124,14/225,40 px; constant velocity — 72,40/143,62 px. Наблюдаемый объект движется вниз, прогноз MolmoMotion в значительной части горизонта уходит выше изображения. Это условный результат одного эпизода, с RGB-proxy соответствиями и неподтверждённой точной active color K. Физическая интерпретация F30 как 2 с предполагает 15 Hz; существующий XYZ-прогноз интерполирован к настоящим RGB 10 Hz. Новые RGB кадры не синтезировались.

Сохранены [данные FMB](data/fmb/), [provenance](data/fmb_provenance.json) и исходный [подробный анализ](../report/fmb_quantitative_2d_episode_5201.md).

## Dobb·E / episode_3651

HoNY `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30`. Для обеих калибровок использован один RGB-префикс 0–95. Вход MolmoMotion — одинаковые RGB 96–98, восемь уже выбранных history-only RGB ID, измеренная HoNY depth, опубликованные Dobb·E c2w labels, оптический базис, действие `Pick up the red cup.`, seed 0 и один checkpoint `allenai/MolmoMotion-4B-H3-F30`, revision `3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`, BF16. Все эти входы зафиксированы хешами до inference.

| Параметр крупнейшей компоненты | A: текущий clean PyCOLMAP | B: новый официальный automatic_reconstructor |
|---|---:|---:|
| Camera model | PINHOLE | PINHOLE |
| fx / fy, px | 1781,69 / 2699,53 | 212,25 / 375,72 |
| Зарегистрировано | 18/96 | 85/96 |
| Sparse points | 101 | 662 |
| Наблюдения sparse tracks | 1208 | 11038 |
| Внутренняя средняя перепроекция, px | 1,152 | 1,243 |

A — существующий masked causal pipeline с DSP-SIFT, affine и guided matching. B — отдельная чистая DB, официальный Windows COLMAP 4.2.1, video/high, single-camera, GPU matching, без масок, ручной K, initial pair и ручных mapper thresholds. Среди реально построенных моделей каждого метода выбрана крупнейшая компонента с ненулевыми 3D-точками; прогноз и будущие кадры не использовались для этого выбора. У B три компоненты: 85, 18 и 10 кадров; уникальное покрытие 93/96. У A четыре компоненты, уникальное покрытие 33/96. Показанные 18/96 и 85/96 относятся к одной крупнейшей компоненте, не к сумме компонент.

Прежний официальный SIMPLE_RADIAL baseline не использован в этом сравнении: его ненулевая distortion стала бы дополнительной переменной. Новый запуск B использует PINHOLE, как A. Оба K переведены из COLMAP в OpenCV: `cx=cy=128−0.5=127.5`. Обе ветви имеют нулевую distortion. Различается только K при unprojection и projection. Позы COLMAP экспортированы как диагностические сведения в `reconstruction.json` и native sparse/TXT; в обеих ветвях MolmoMotion использованы одинаковые Dobb·E labels, без COLMAP extrinsics.

Отдельно сохранены [история A (3,8,3)](data/dobbe/A/history.npy), [история B (3,8,3)](data/dobbe/B/history.npy), [прогноз A (8,30,3)](data/dobbe/A/prediction.npy), [прогноз B (8,30,3)](data/dobbe/B/prediction.npy), K, raw model outputs, receipts, общий UV/depth/poses, полные camera trajectories и sparse statistics. Оба raw ответа независимо разобраны официальным parser: 240/240 позиций, точное совпадение текста с массивами.

В видео на обоих половинах используется один и тот же реальный RGB кадр 99–158, зелёные наблюдения одинаковы. Прогнозы интерполированы с предполагаемых 15 Hz к настоящим RGB 30 Hz; их projection использует одинаковые опубликованные poses каждого будущего кадра. RGB tracking выполняется без K и прогнозов, сохраняет прежние ID, исключает потерянные или ненадёжные наблюдения: принято 192/480, поэтому это частичный RGB proxy. Наблюдаемые траектории нарисованы отдельно для каждого ID, чтобы смена доступного набора не создавала ложный скачок среднего. Оранжевая линия в RGB — среднее прогнозируемых точек. Все исходные координаты проекций сохранены, без обрезки до границ RGB; внекадровые позиции явно подписаны.

Внизу обеих половин показаны три history позиции и все 30 forecast позиций каждой из восьми точек в одной системе камеры t0, с одинаковыми пределами и ракурсом. Ещё не достигнутая часть горизонта показана бледной линией. PNG использует ранний кадр 104, t0+0,20 с: все восемь наблюдаемых ID доступны и различие проекций хорошо читается. Это выбор кадра для представления результата после rendering, без изменения экспериментальных входов.

| Расхождение A/B | Значение |
|---|---:|
| Средняя Δ history по 24 позициям | 38,88 мм |
| Средняя Δ forecast по 240 позициям | 39,39 мм |
| Δ forecast на последнем шаге | 39,40 мм |
| Средняя Δ перемещения относительно собственного t0 каждой точки | 5,17 мм |
| Средняя Δ projected forecast по 30 срокам | 2398,14 px |
| Δ projected forecast на последнем сроке | 5028,85 px |

Большая часть абсолютного 3D-различия уже присутствует во входных историях; изменение прогнозируемого перемещения значительно меньше. Огромные различия проекций связаны также с разной K при движущейся камере и сохраняются за границами RGB. Эти числа характеризуют чувствительность всего unprojection→forecast→projection pipeline, а не только изменение предсказываемого движения модели.

Обе K остаются неподтверждёнными оценками. Проверка на одинаковых 22 автоматических RGB-соответствиях с measured depth и labels дала median reprojection 75,05 px для A и 10,27 px для B; меньшее число B само по себе не сертифицирует калибровку. Depth-Z/ray и качество labels остаются ограничениями. В labels есть скачок 156→157 на 10,49 см; он одинаково используется в обеих ветвях и подписан в соответствующей части видео. Это не оценка победителя по полноценному GT.

Все численные данные: [metrics](data/dobbe/comparison_metrics.json), [покадровые дельты](data/dobbe/comparison_by_step.csv), [geometry audit](data/dobbe/geometry_audit.json), [provenance](data/dobbe_provenance.json). Исходные native sparse models, DB, логи и TXT находятся в `runs/dobbe_two_colmap_v1/official` и прежнем `runs/dobbe_colmap_clean_rerun_v1`.

## Проверка и воспроизведение

[Artifact audit](data/artifact_audit.json) проверяет полное декодирование двух H.264 MP4, PNG, формы history/forecast, реальные parser outputs, хеши frozen inputs, идентичность остальных параметров A/B и сохранённых копий массивов. Скрипты не изменяют прежние исследовательские результаты.

```powershell
cd F:\AIRI_task\molmo-motion
.\scripts\run_dobbe_two_colmap.ps1
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/dobbe_two_colmap.py infer
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/check_two_colmap_geometry.py
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/build_motion_visualizations.py all
```

Готовые successful reconstruction receipts и predictions повторно используются; visualizations и их аудит пересобираются. Требуются существующая локальная WSL `.venv`, RTX 4070, checkpoint и исследовательские данные. [CLI COLMAP](https://colmap.github.io/cli.html) и [модели камер / pixel convention](https://colmap.github.io/cameras.html) описывают официальный интерфейс и отличие PINHOLE от SIMPLE_RADIAL.
