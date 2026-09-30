# Диагностика подготовки

Проверка выполнена 30 сентября 2026 года на Windows 11 и Ubuntu 24.04.3 LTS
в WSL2. В таблице указано начальное состояние; итоговые измерения приведены
ниже.

| Компонент | Обнаружено |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, 12 282 MiB VRAM, compute capability 8.9 |
| NVIDIA driver | 591.86; `nvidia-smi` работает в Windows и WSL |
| RAM хоста | 34 088 603 648 байт (31,75 GiB) |
| Диск F до установки | 164 620 247 040 байт свободно (153,32 GiB) |
| Остальные диски до установки | C: 181 521 969 152; D: 49 352 839 168; E: 2 559 946 752 байт свободно |
| Git | 2.54.0.windows.1 |
| Windows Python | 3.12.3, `F:\Apps\Python\python.exe`; torch там не установлен |
| Windows Miniconda | 25.7.0 в `F:\Apps\miniconda3`; существующие среды не изменялись |
| WSL | Ubuntu 24.04.3, kernel 6.6.87.2-microsoft-standard-WSL2 |
| Лимит WSL после настройки | 24 GB RAM, 6 GiB swap |
| Изолированный Python | 3.11.16 в `F:\AIRI_task\.venv` |
| PyTorch | 2.9.1+cu128; `torch.version.cuda == 12.8` |
| MolmoMotion source | `61f5b21b694ad8f854ec7ecd2400005acc73f685` |

Изначально WSL был в режиме NAT и не подключался к `pypi.org` или GitHub по
HTTPS. В Windows сеть работала. Режим `networkingMode=mirrored` устранил
ошибку: из WSL получен HTTP 200 от PyPI. Лимит RAM WSL был увеличен с
примерно 16 до 24 GB, так как один `model.pt` имеет размер 17,85 GB.

`python scripts/check_setup.py` прошёл: PyTorch импортировался, сообщил
`torch.cuda.is_available() == True`, небольшая CUDA-операция дала правильный
результат, классы `MolmoMotion` и `MolmoMotionProcessor` импортировались.
OpenCV, PyAV и Decord декодировали вложенное видео
`data_generation/third_party/vipe/assets/examples/dog-example.mp4`;
Decord и TorchCodec увидели по 122 кадра, первый кадр OpenCV имеет размер
720 × 1280 × 3. BF16 поддерживается видеокартой. TorchCodec 0.9.1 взят в
CPU-варианте, для него в WSL установлены общие библиотеки FFmpeg 6.1
(`libavdevice60`, `libavfilter9`, `libswscale7`, `libswresample4` и их
зависимости). CUDA-вариант TorchCodec требовал бы также NVIDIA NPP, тогда
как видео здесь декодируется на CPU.

Авторская команда `python examples/01_quickstart.py --from-prediction
--output runs/preview_from_bundled_prediction.mp4` прошла и сохранила MP4
(90 865 байт). Она использовала предсказание из репозитория, поэтому **не
является локальным inference MolmoMotion**.

Проверка `python -m pip check` выдаёт одно предупреждение:
`decord 0.6.0 is not supported on this platform`. Опубликованный wheel
`decord 0.6.0` помечен тегом `cp36-cp36m-manylinux2010_x86_64`, хотя
фактический импорт и декодирование в Python 3.11 здесь проходят. Других
конфликтов проверка не сообщила. Метаданные пакета вручную не исправлялись.

Для первого запуска выбран только авторский checkpoint
`allenai/MolmoMotion-4B-H3-F30`, revision
`3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`: `config.yaml` и
`model.pt` (17 847 894 425 байт). Четыре альтернативных `safetensors`-шарда,
MolmoMotion-1M и полный PointMotionBench не загружаются. Для входа выбран
вложенный пример `examples/data/davis_bmx_trees` с тремя кадрами и готовыми
входными 2D/3D точками. Для последующего сравнения с реальным продолжением
потребуется отдельно получить короткий соответствующий DAVIS-эпизод.

## Итоговый smoke test

`python scripts/smoke_inference.py` загрузил `model.pt` в BF16 и выполнил
настоящий `predict_trajectory()` по трём историческим кадрам встроенного
эпизода, восьми входным точкам и описанию действия. Прогноз одного будущего
кадра имеет форму `(8, 1, 3)` и конечные значения. Файлы результата:
`runs/smoke_real_f1.npz` (220 байт) и `runs/smoke_real_f1.json`.

| Измерение | Значение |
| --- | ---: |
| Время от начала скрипта после импортов | 68,35 с |
| Пиковая выделенная VRAM PyTorch | 9,42 GiB |
| Пиковая зарезервированная VRAM PyTorch | 9,86 GiB |
| Пиковый RSS процесса Linux | 22,32 GiB |
| Свободно на F: после подготовки | 138 304 131 072 байт (128,80 GiB) |
| Дополнительно занято на F: | 26 316 115 968 байт (24,51 GiB) |
| Дополнительно занято на C: из-за WSL | около 1,90 GiB |

RSS включает отображённые через `mmap` страницы checkpoint и не равен
объёму анонимной RAM. На F: среда `.venv` занимает около 7,6 GiB,
checkpoint около 17 GiB, управляемый Python около 92 MiB.

После этого из нового процесса WSL по командам `README_SETUP.md` была
активирована `.venv` и повторно выполнен `python scripts/check_setup.py`:
импорты, CUDA-операция и чтение видео прошли. Полный 30-кадровый прогноз,
сравнение с эталоном, ShareRobot и DaS относятся к следующему этапу.
