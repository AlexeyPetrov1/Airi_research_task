# FMB wrist v4: воспроизведение

Этот run продолжает исследование одного эпизода FMB/ShareRobot 5201. Два ракурса используют один TCP stream. Результат исследовательский: будущее этого эпизода уже было просмотрено в v3. Калибровка, 24 ID и reference унаследованы из v3; новые параметры выбраны только по observed frames 77–126.

## Проверить сохранённый результат

Команды выполняются в Ubuntu WSL из `/mnt/f/AIRI_task/molmo-motion` с Python `/mnt/f/AIRI_task/.venv/bin/python`.

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_audit.py
```

Аудитор сверяет сохранённые raw model outputs с parser и tensors, исходные RGB/XYZ и порядок ID, воспроизводит robot forecast из past TCP, проверяет прежние reference masks и метрики baseline, видео, SHA256 просмотренных изображений и всех сохранённых файлов v3. Он обновляет только `completion_audit.json` этого run; исходный v3 не изменяется.

Пять содержательных геометрических тестов проверяют body-frame rotation, damping, rigid attachment, moving-camera projection, смену base frame и обработку невалидных centroid points:

```bash
/mnt/f/AIRI_task/.venv/bin/python -m pytest tests/test_fmb_wrist_v4.py -q
```

## Повторить эксперимент в новой папке

Не запускайте подготовку поверх завершённого run: frozen inputs защищены от перезаписи. Для независимого повторения используйте отдельную копию репозитория и задайте новый `OUT` в `scripts/fmb_wrist_v4_diagnose.py`; `BASE` должен указывать на неизменённый `runs/fmb_wrist_v3`. Report writer также содержит фиксированные имена ссылок и пути отчёта: замените их в этой копии на имя нового run. Существующие v3/v4 сохраняйте.

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_diagnose.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_sources.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_prepare.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_refine.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_infer.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_evaluate.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_sensitivity.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_v4_media.py
```

Inference требует CUDA, BF16 и уже доступных официальных checkpoint в `data/checkpoints/MolmoMotion-4B-H3-F30` и `data/checkpoints/MolmoMotion-4B-H1-F32`. Использованные revisions записаны в каждом `group_XX/model_run.json`. Выполняются 12 успешных P8 calls: два checkpoint × две камеры × три группы. У H3 сохраняются все 30 шагов, у H1 — все 32; оценка использует общие 0,1–2,0 секунды с интерполяцией XYZ с 15 на 10 Hz.

При штатном прерывании `--resume` принимает только завершённые группы с проверенными hashes, anchor и строгим parser. Незавершённую попытку сначала явно сохраните отдельно и разберите причину; автоматического удаления либо выбора лучшей попытки нет. В текущем run сохранены одна дополнительная прерванная генерация и две ошибки инициализации до генераций.

Визуальная оценка выполняется фактическим встроенным просмотром PNG, без внешней HF VLM. `builtin_visual_review.json` содержит пути, SHA256 и наблюдения. Не переносите эту запись на изображения нового run: их нужно просмотреть заново. После реального просмотра сохраните новый receipt и результат тестов, затем сформируйте отчёт и запустите аудит. Receipt подтверждает работу ассистента; скрипт не выполняет визуальную оценку автоматически.

Сохранённый completion auditor проверяет также историю именно текущего run: initial code snapshot, одну прерванную генерацию и две ошибки инициализации. Чистый повтор не обязан воспроизводить эти ошибки. Для аудита нового run адаптируйте проверки его фактического журнала и пути отчёта; не копируйте записи старых попыток как свидетельство новых вызовов.

## Что воспроизводится

Robot forecast использует последние три past TCP poses и damping tau=0,8 s, выбранные на observed pseudo-futures. Fixed X преобразует rigid t0 template в общую `camera_t0`. Future TCP нужен только reference и проекции для оценки, а не forecaster. Гибрид добавляет общий median displacement исходного MolmoMotion, ограниченный observed p90 camera-relative drift: 2,41 мм для wrist_2 и 3,23 мм для wrist_1.

Семь bootstrap X из прежнего observed fit меняют только estimated reference для описательной проверки ранжирования. Это не доверительный интервал и не совместная неопределённость reference/forecast. Для проверки обобщения нужен новый эпизод с заранее замороженным протоколом.
