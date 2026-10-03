# MolmoMotion: воспроизводимые эксперименты

Семь фиксированных примеров AIRI с общим CLI, метриками и визуализацией. Упаковка сохраняет поведение commit `70ed2c4`; алгоритмы и vendored-код [MolmoMotion](https://github.com/allenai/molmo-motion) не изменены.

| Конфиг | Пример |
|---|---|
| `author_davis` | DAVIS BMX, авторский H3/F30, 8 точек |
| `fmb_wrist_1`, `fmb_wrist_2` | Две FMB камеры, выбранная кинематика + ограниченная поправка MolmoMotion |
| `berkeley_bottle` | Сохранённые bottle-варианты; основной `late_036_lift005` |
| `berkeley_cup` | Основной `original_physical` |
| `dobbe` | Dobb-E `pure_vipe`, условная 2D диагностика |
| `dobbe_blocked` | Ожидаемая остановка hybrid по `static_median` до inference |

Проверяемая среда: Linux/WSL2, Python 3.11, RTX 4070 12 GB. Точный снимок зависимостей: [configs/installed-packages-wsl.txt](configs/installed-packages-wsl.txt). Для TorchCodec нужны системные библиотеки FFmpeg.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec==0.9.1 --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install '.[dev]'
hf download allenai/MolmoMotion-4B-H3-F30 config.yaml model.pt \
  --revision 3f5e790a511ff2cdf21c8d2a14cb4d8409c94629 \
  --local-dir data/checkpoints/MolmoMotion-4B-H3-F30
```

Веса занимают около 18 GB. Immutable-наборы `fixtures/` содержат точные наблюдаемые входы, отдельные evaluation-данные и единственную копию прогнозов для replay. RGB хранится lossless; большие исходные датасеты и повторная подготовка геометрии не нужны. Wheel включает конфиги и fixtures.

Новый inference (BF16, штатный greedy decoder, seed 0):

```bash
molmo-motion-experiment --config configs/fmb_wrist_1.json \
  --checkpoint /absolute/path/to/MolmoMotion-4B-H3-F30
```

Эквивалент: `python -m motion_experiments.run ...`. Команда доступна из любого каталога; относительные пути конфигов разрешаются внутри установленного пакета. Для 24 точек выполняются три P8-вызова; метрики используют все точки, основное видео показывает выбранные восемь.

Replay всех примеров:

```bash
molmo-motion-experiment --config configs/author_davis.json configs/fmb_wrist_1.json \
  configs/fmb_wrist_2.json configs/berkeley_bottle.json configs/berkeley_cup.json \
  configs/dobbe.json configs/dobbe_blocked.json --mode replay \
  --checkpoint /absolute/path/to/MolmoMotion-4B-H3-F30
```

Результаты создаются в уникальном `outputs/<run_id>/`: откройте `index.html`. Сохраняются входы, протокол, прогнозы, метрики, ошибки, 2D/3D координаты, графики и видео. Можно указать `--output /absolute/path/to/new/run`; существующие каталоги и fixtures защищены от записи. Для повторной отрисовки свежего завершённого inference добавьте `--mode replay --predictions-from /absolute/path/to/inference_run`.

Полная проверка после изменений из checkout:

```bash
MOLMO_CHECKPOINT=/absolute/path/to/MolmoMotion-4B-H3-F30 pytest -q
python tools/verify_final.py --checkpoint /absolute/path/to/MolmoMotion-4B-H3-F30 \
  --baseline-run /absolute/path/to/baseline_replay --python /absolute/path/to/installed/env/bin/python
```

Verifier запускает тесты, все семь replay, новый inference всех шести успешных примеров, отрицательный gate и `--predictions-from`; сравнивает результаты с baseline и пишет [docs/final_verification.json](docs/final_verification.json) и [docs/final_verification.md](docs/final_verification.md). Без нового inference задача считается FAIL. Требования: [docs/GOAL.md](docs/GOAL.md), происхождение: [docs/provenance.json](docs/provenance.json), размеры: [docs/packaging_report.md](docs/packaging_report.md).

DAVIS использует общую систему первой камеры; покадрового 2D ADE/FDE нет. FMB depth/hand-eye и Berkeley K оценены, поэтому 3D reference условный. Bottle postprocessing и визуальный выбор относятся к прежнему исследованию. Частоты истории сохранены: DAVIS 24 Hz, FMB 10 Hz, Berkeley 5 Hz при обучении на 15 Hz. Dobb-E не имеет подтверждённого 3D GT; hybrid отвергается по median static reprojection 5.524 px > 4 px. Для Dobb-E исходный полный processor packet отсутствовал: сохраняется точный input IDs SHA-256 и прежнее свидетельство camera/world equivalence; для остальных примеров сравниваются точные типы, формы, dtype и SHA-256 всех полей.
