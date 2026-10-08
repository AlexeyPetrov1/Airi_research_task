<p align="center">
  <img src="assets/readme/cover.png" alt="Диффузионная генерация видео для робототехнических сцен: нейросетевая 3D-геометрия, прогноз траекторий и видеосинтез" width="1200">
</p>

# Диффузионная генерация видео для робототехнических сцен

**MolmoMotion × Diffusion as Shader · AIRI · Петров Алексей [@aapetrov23](https://t.me/aapetrov23)**

Использовал робототехнические сцены **Berkeley Autolab UR5 из ShareRobot**, восстановил 3D-геометрию с помощью нейросетевой оценки параметров камеры и измеренной глубины, предсказал движение точек в MolmoMotion и передал траекторию в диффузионную модель DaS. Итоговый вариант **H5** генерирует видео с движением стакана и манипулятора.

**RGB-D → нейросетевая 3D-геометрия → прогноз 3D-траекторий → диффузионное видео.**

| Прогноз движения | Генерация видео | Оборудование |
|---|---|---|
| MolmoMotion-4B-H3-F30 · BF16 | DaS / Wan2.1-Fun 1.3B Control | RTX 4070, 12 ГБ VRAM · 32 ГБ RAM · Linux / WSL |

[3D-геометрия](#geometry) · [Предсказание траекторий](#trajectories) · [Генерация H5](#h5) · [Воспроизведение](#reproduce)

### Реальная сцена и сгенерированное видео

| Полная реальная запись Berkeley UR5 | Диффузионная генерация H5 |
|:---:|:---:|
| [![Полная запись эпизода 10 Berkeley UR5 с исходной скоростью](assets/readme/berkeley-real.gif)][video-real] | [![DaS H5: движение стакана, захвата, плеча и предплечья](assets/readme/das-final.gif)][video-h5] |
| Все 120 кадров, 24 секунды, исходная частота 5 кадров/с. | 49 кадров, около 6 секунд; дуга прогноза растянута с 2 до 6 секунд. |

*GIF открываются по клику как исходные MP4. Реальная запись показывает весь эпизод; H5 показывает продолжение после t₀ = 63. Обложка служит иллюстрацией проекта; результаты эксперимента показаны в GIF и таблицах.*

<a id="geometry"></a>
## 1. Восстановление 3D-геометрии

В Berkeley выбрал перенос бутылки, **эпизод 9, t₀ = 47**, и стакана, **эпизод 10, t₀ = 63**. Для подготовки входа использовал кадры до t₀. В TFDS нашёл глубину RealSense, совмещённую с RGB; параметры камеры **K оценил с помощью UniDepthV2**. Неподвижность камеры проверил по фону.

<p align="center">
  <img src="assets/readme/berkeley-pipeline.jpg" alt="Berkeley: RGB-D, нейросетевая оценка камеры, MolmoPoint, SAM 2.1, AllTracker и подготовка 3D-истории" width="1100">
</p>

Последовательность подготовки:

1. **MolmoPoint** находит предмет на кадре t₀.
2. **SAM 2.1** выделяет маску предмета.
3. **AllTracker** отслеживает точки на предыдущих кадрах.
4. Из 100 кандидатов выбираю **24 точки**, прошедшие проверку.
5. По пикселям, глубине и K строю историю 3D-координат.

Глубину точки брал как медиану окна **5 × 5 пикселей**. При сглаживании менял глубину, сохраняя положение точки на изображении. Геометрия остаётся приближённой: нейросеть оценивает параметры камеры, а измеренная глубина задаёт расстояния. Обозначение **3D_est** в метриках отражает эту неопределённость.

<a id="trajectories"></a>
## 2. Предсказание 3D-траекторий

**MolmoMotion-4B-H3-F30** получает три RGB-кадра, историю 3D-точек и текстовое описание действия. За один запуск предсказывает движение **8 точек на 30 шагов**; для 24 точек выполнял три запуска. Модель не дообучал; расчёты выполнял в BF16 и проверял полноту ответа.

Для стакана использовал кадры **61, 62, 63**: моменты −0,4, −0,2 и 0 с относительно t₀. Исходная запись имеет частоту **5 кадров/с**, а модель рассчитана на **15 кадров/с**. Для сравнения с реальным продолжением взял шаги 3, 6, …, 30, соответствующие 0,2, 0,4, …, 2,0 с. Влияние различия частот отдельно не проверял.

| Перенос бутылки | Перенос стакана |
|:---:|:---:|
| [![Berkeley, бутылка: прогноз и наблюдаемое движение](assets/readme/berkeley-bottle.gif)][video-bottle] | [![Berkeley, стакан: прогноз и наблюдаемое движение](assets/readme/berkeley-cup.gif)][video-cup] |

В этих сценах модель предсказывала слишком большое перемещение. Проверил уменьшение смещения относительно t₀: для бутылки сохранён вариант **`late_036_lift005`**, для стакана показана поправка **×1/3**. Коррекцию выбирал на уже просмотренных сценах. Для генерации H5 сохранил **исходную дугу стакана**, без коррекции ×1/3.

| Сцена | Прогноз | ADE 2D, px ↓ | FDE 2D, px ↓ | ADE 3D_est, мм ↓ |
|---|---|---:|---:|---:|
| Бутылка | Исходный | 218,39 | 201,40 | 266,49 |
| Бутылка | После коррекции, `late_036_lift005` | 45,69 | 33,58 | 58,91 |
| Бутылка | Постоянная скорость | 11,05 | 11,93 | 42,62 |
| Стакан | Исходный | 193,57 | 224,48 | 222,20 |
| Стакан | Смещение ×1/3 | 46,92 | 69,21 | 58,52 |
| Стакан | Постоянная скорость | 2,91 | 7,37 | 14,29 |

ADE измеряет среднюю ошибку по доступным парам «точка / кадр», FDE измеряет ошибку на последнем шаге. В обеих сценах постоянная скорость точнее по этим метрикам. Они не учитывают столкновения и сами по себе не проверяют выполнение действия. Для 2D доступны **240/240 пар** в каждой сцене; для 3D_est **235/240** у бутылки и **238/240** у стакана.

<details>
<summary><b>Исходные траектории и результат коррекции</b></summary>

![Исходные 3D-траектории Berkeley: входная история и предсказанные пути](assets/readme/berkeley-original.jpg)

![Траектории Berkeley после уменьшения предсказанного смещения](assets/readme/berkeley-scaled.jpg)

Розовым показан исходный прогноз, голубым обозначена входная история. В итоговом наборе основным вариантом бутылки выбран `late_036_lift005`, а стакана `original_physical`.

</details>

[История Berkeley](runs/berkeley_ur5_molmomotion/research_story.md) · [сохранённые прогнозы и метрики](https://github.com/AlexeyPetrov1/Airi_research_task/blob/f7403e5e5685fe8d235ad7565d8315804b4e0e11/docs/experiments.md).

<a id="h5"></a>
## 3. H5: диффузионная генерация видео

Использовал **DaS из ветки Wanfun с Wan2.1-Fun 1.3B Control**. DaS получает кадр t₀ и управляющее видео, построенное по сохранённому прогнозу MolmoMotion. Будущие RGB-кадры в H5 использовал только для оценки после генерации.

<p align="center">
  <img src="assets/readme/das-pipeline.svg" alt="MolmoMotion → управляющее видео → DaS H5: подготовка, генерация и результат" width="1100">
</p>

Для управления выбрал **первую группу из восьми точек**: её траектории лучше соответствовали движению одного жёсткого тела. Цвет каждой точки сохранял во всех кадрах. Исходную пространственную дугу растянул с **2 до 6 секунд**.

Движение плеча и предплечья UR5 восстановил приближённо по состояниям суставов до t₀. На исходном кадре выделил части манипулятора, восстановил скрытый фон и подготовил их движение. Траектории стакана и захвата сохранил. Опорное видео из t₀ использовал после каждого шага генерации для ограничения внешнего вида с коэффициентом **0,25**.

[![Управляющее видео H5: движение стакана и звеньев манипулятора](assets/readme/das-control.gif)][video-control]

*Управляющее видео H5. Один и тот же цвет обозначает одну и ту же точку поверхности во всех кадрах.*

| Параметр H5 | Значение |
|---|---|
| Кадры и длительность | 49 кадров, около 6 секунд |
| Генерация | Seed 42, 30 шагов, BF16 и перенос частей модели в RAM |
| Время генерации | 507,4 с |
| Пик выделенной VRAM / памяти процесса | 9,88 GiB / 21,14 GiB |
| ADE / FDE к выбранному прогнозу | 4,31 / 5,06 px |
| Доступные пары | 281/392, или 71,68% |

Метрики H5 сравнивают генерацию с **выбранной управляющей траекторией**. Потерянные трекером точки в оценку не входят. Просмотрел все 49 кадров: стакан остаётся узнаваемым, хотя рисунок меняется. Приближённая геометрия манипулятора не подходит для управления реальным роботом.

### Итоговое видео H5

<p align="center">
  <a href="https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/eb3a5240140386b021b5555b8e0b3bde3ee60b0f/runs/berkeley_ur5_molmomotion/cup/das_full_motion/H5_group00_6s_whole_robot_prior025/generated_seed42.mp4">
    <img src="assets/readme/das-final.gif" alt="Итог H5: стакан, захват, плечо и предплечье движутся по заданной дуге" width="900">
  </a>
</p>

В H5 движутся **стакан, захват, плечо и предплечье**; основание остаётся неподвижным. DaS воспроизводит выбранную дугу. Стакан заканчивает движение выше и правее целевого стакана и не попадает внутрь него.

<a id="reproduce"></a>
## Воспроизведение

Прогнозы Berkeley и общий CLI находятся в ветке **[`molmo-motion-packaged`](https://github.com/AlexeyPetrov1/Airi_research_task/tree/molmo-motion-packaged)**. Код и материалы генерации H5 находятся в **[`codex/das-full-motion-20261003`](https://github.com/AlexeyPetrov1/Airi_research_task/tree/codex/das-full-motion-20261003)**; для DaS требуется отдельная среда.

<details>
<summary><b>Установка и запуск прогноза Berkeley в Linux / WSL</b></summary>

```bash
git clone --single-branch --branch molmo-motion-packaged \
  https://github.com/AlexeyPetrov1/Airi_research_task.git
cd Airi_research_task
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec==0.9.1 \
  --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install '.[dev]'
hf download allenai/MolmoMotion-4B-H3-F30 config.yaml model.pt \
  --revision 3f5e790a511ff2cdf21c8d2a14cb4d8409c94629 \
  --local-dir data/checkpoints/MolmoMotion-4B-H3-F30

# Новый прогноз стакана
molmo-motion-experiment --config configs/berkeley_cup.json \
  --checkpoint "$PWD/data/checkpoints/MolmoMotion-4B-H3-F30"

# Метрики и визуализация сохранённого прогноза
molmo-motion-experiment --config configs/berkeley_cup.json --mode replay \
  --checkpoint "$PWD/data/checkpoints/MolmoMotion-4B-H3-F30"
```

Для бутылки используйте `configs/berkeley_bottle.json`. Нужны системные библиотеки FFmpeg для TorchCodec; веса занимают около 18 ГБ. Результаты сохраняются в `outputs/<run_id>/`, где `index.html` собирает прогнозы, метрики, графики и видео.

</details>

Работа основана на [MolmoMotion от Ai2](https://github.com/allenai/molmo-motion), исходный commit `61f5b21b694ad8f854ec7ecd2400005acc73f685`. [Веса модели](https://huggingface.co/allenai/MolmoMotion-4B-H3-F30) · [лицензия основного кода](LICENSE) · [происхождение визуальных материалов](assets/readme/sources.json). Датасеты и сторонние компоненты имеют собственные условия использования.

[video-real]: https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/eb3a5240140386b021b5555b8e0b3bde3ee60b0f/runs/berkeley_ur5_molmomotion/sources/videos/chunk-000/observation.images.image/episode_000010.mp4
[video-h5]: https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/eb3a5240140386b021b5555b8e0b3bde3ee60b0f/runs/berkeley_ur5_molmomotion/cup/das_full_motion/H5_group00_6s_whole_robot_prior025/generated_seed42.mp4
[video-bottle]: https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/f7403e5e5685fe8d235ad7565d8315804b4e0e11/fixtures/berkeley_bottle/media/legacy_comparison.mp4
[video-cup]: https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/f7403e5e5685fe8d235ad7565d8315804b4e0e11/fixtures/berkeley_cup/media/legacy_comparison.mp4
[video-control]: https://raw.githubusercontent.com/AlexeyPetrov1/Airi_research_task/eb3a5240140386b021b5555b8e0b3bde3ee60b0f/runs/berkeley_ur5_molmomotion/cup/das_full_motion/group00_stretched_6s_whole_robot_v1/control_720x480.mp4
