# MolmoMotion: финальные воспроизводимые эксперименты

Отдельный минимальный репозиторий для пунктов 1–2 задания AIRI. История исследования остаётся в соседнем `molmo-motion`; старые результаты не перезаписываются. Код модели скопирован из проверенного исходного репозитория без изменений: [Ai2](https://github.com/allenai/molmo-motion), Apache 2.0. Контрольные суммы находятся в `docs/model_source_snapshot.json`.

Выбранные примеры:

| Конфиг | Что воспроизводится |
|---|---|
| `configs/author_davis.json` | Велосипед: DAVIS `bmx-trees`, H3/F30, 8 точек, реальные кадры 3–32 |
| `configs/fmb_wrist_1.json` | Нижняя левая панель указанного FMB видео: выбранная кинематика + ограниченная поправка MolmoMotion |
| `configs/fmb_wrist_2.json` | Тот же FMB эпизод, вторая камера; собственные исходные точки |
| `configs/berkeley_bottle.json` | Три варианта `timing_group_02.mp4`; основной — `late_036_lift005`, показана группа 02 |
| `configs/berkeley_cup.json` | Левая панель `fixed_comparison.mp4`: `original_physical` |
| `configs/dobbe.json` | `C/pure_vipe`: реальный прогноз, условная 2D диагностика |
| `configs/dobbe_blocked.json` | Реальная ветка `C/hybrid`: `SKIPPED_GEOMETRY_GATE`, причина `static_median` |

## Установка

Проверяемая среда: Linux/WSL2, Python 3.11, PyTorch 2.9.1+cu128, RTX 4070 12 GB. Создайте отдельную среду и установите пакет:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install 'torchcodec==0.9.1' --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install -e '.[dev]'
```

Для TorchCodec нужны системные библиотеки FFmpeg. Точный снимок исходной рабочей среды: `configs/installed-packages-wsl.txt`. Прогноз использует BF16 и штатный greedy decoder. Модель не изменена, параметры по будущим данным не подбираются.

## Данные и checkpoint

`data/legacy/` содержит небольшую автономную копию подготовленных данных выбранных эпизодов: RGB history/future, исходные XYZ/UV, камеры, ID, прогнозы, метрики, сырой ответ модели и пакет процессора. Большие исходные датасеты не нужны для повторения этих фиксированных экспериментов. Это адаптеры уже подготовленных native-артефактов, а не повторный запуск ViPE/AllTracker/калибровки. Происхождение подготовки описано в `docs/legacy_inventory.md` и `docs/borrowed_functions.json`.

`tests/golden/source/` — независимые копии прежних результатов, сохранённые **до** адаптации. `tests/golden/manifest.json` содержит SHA-256 и ожидаемый отрицательный статус. Эти файлы не пересоздаются новым pipeline.

Для новой отрисовки DAVIS отдельно добавлены исходные RGB будущих кадров, без старых нарисованных точек. Они побайтно совпадают с декодированными кадрами прежнего эксперимента; происхождение и контрольные суммы — в `docs/evaluation_rgb_source.json`.

Дополнительный старый подробный отчёт Berkeley зафиксирован в `docs/supplemental_reference_manifest.json`. Проверяются также прежние median/P90, PWT, ошибки по точкам, baseline-проекции и показатели формы; существующие эталоны не перезаписываются.

Для запуска модели скачайте **только** нативный checkpoint:

```bash
hf download allenai/MolmoMotion-4B-H3-F30 config.yaml model.pt \
  --revision 3f5e790a511ff2cdf21c8d2a14cb4d8409c94629 \
  --local-dir data/checkpoints/MolmoMotion-4B-H3-F30
```

Веса занимают около 18 GB. На данной машине можно использовать уже скачанный checkpoint через `--checkpoint ../molmo-motion/data/checkpoints/MolmoMotion-4B-H3-F30`; код и данные нового репозитория самостоятельны. При конкурирующем крупном вычислении inference ожидает освобождения памяти, не прерывая другие процессы.

## Один способ запуска

Из любого каталога после `pip install -e .`:

```bash
molmo-motion-experiment --config configs/fmb_wrist_1.json
molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json
molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json
molmo-motion-experiment --config configs/author_davis.json
```

Эквивалент: `python -m motion_experiments.run ...`. Относительные пути конфигов разрешаются от корня репозитория. По умолчанию выполняется **новый inference**. Три P8-вызова дают все 24 точки FMB/Berkeley; все точки участвуют в метриках, 8 выбранных — в основном видео.

Быстрая проверка вычислений и новой визуализации на прежнем прогнозе:

```bash
molmo-motion-experiment --config configs/author_davis.json configs/fmb_wrist_1.json \
  configs/fmb_wrist_2.json configs/berkeley_bottle.json configs/berkeley_cup.json \
  configs/dobbe.json configs/dobbe_blocked.json --mode replay
```

Режим `replay` проверяет процессор и использует сохранённый neural forecast; новый вызов модели он не доказывает. Режим и `fresh_inference` записываются во все результаты. Новые запуски всегда получают отдельный каталог `outputs/<run_id>/`; существующий каталог не перезаписывается.

Чтобы после нового inference повторить оформление без повторной GPU-генерации, добавьте к `--mode replay` аргумент `--predictions-from outputs/<inference_run>`. Проверяются завершённый статус, SHA-256 нового прогноза и одинаковые наблюдаемые входы. Источник прогноза явно записывается в metadata.

## Результаты и проверка

Откройте `outputs/<run_id>/index.html`. В каждом примере: `inputs/canonical.npz`, точные пакеты процессора, `prediction_parity.json`, `metrics.json`, ошибки NPZ, прогноз и сравнение с реальным продолжением MP4, точки на изображении, полные 2D/3D траектории, XYZ по времени, ADE/FDE и графики 2D/3D ошибок. `render_receipt.json` проверяет декодированное число кадров и сохраняет координаты для проверки отрисовки.

```bash
pytest -q
```

Тесты сравнивают исходные входы, кадры, время, геометрию, прогнозы, static/CV, старые метрики и отрицательную ветку Dobb-E. Для DAVIS/FMB/Berkeley все поля сохранённых пакетов процессора сравниваются точно. Dobb-E не сохранил старый полный пакет: проверяются точный SHA-256 сериализованных input IDs и сохранённая проверка эквивалентности camera/world; полноту старого пакета подтвердить нельзя. Прогнозы сравниваются с `atol=1e-7 m`, проекции — с явно указанными допусками. Без установленного checkpoint config тесты процессора обозначены как skipped; для полной проверки config должен быть установлен.

## Существенные ограничения

DAVIS сохраняет авторский пример с XYZ в общей системе первой камеры. Покадровых extrinsics нет, поэтому прогноз показан отдельно в 3D и диагностически в первой камере; достоверная 2D ошибка не заявляется. FMB использует оценённый depth scale и hand-eye, Berkeley — оценённый/candidate K; их 3D reference условный. Улучшения FMB включают роботную кинематику. Bottle сохраняет прежний исследовательский postprocessing и визуальный выбор, поэтому его нельзя выдавать за независимый blind test.

Сохранены и прежние частоты истории: DAVIS 24 Hz, FMB 10 Hz, Berkeley 5 Hz при обучении checkpoint на 15 Hz. Графики показывают физическое время исходного видео; соответствие шагов и прежняя интерполяция прогноза воспроизводятся без нового пересэмплинга истории.

Dobb-E `pure_vipe` действительно прошёл прежний estimated geometry gate, но не имеет подтверждённого 3D эталона. Его 2D сравнение зависит от гипотезы поз и K. Отдельный `hybrid` действительно отвергнут: median static reprojection **5.524 px > 4 px**. Для него pipeline сохраняет причину остановки и не создаёт прогноз или метрики.
