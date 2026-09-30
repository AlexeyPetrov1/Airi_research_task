# Подготовка MolmoMotion для задания AIRI

Проверено 30 сентября 2026 года. Рабочий репозиторий находится в
`F:\AIRI_task\molmo-motion`, исходный код Ai2 — commit
`61f5b21b694ad8f854ec7ecd2400005acc73f685`. Это подготовка к первому
эксперименту: ShareRobot, DaS и исследовательские прогоны здесь не установлены.

## Где что находится

| Путь | Назначение |
| --- | --- |
| `F:\AIRI_task\molmo-motion` | Git-репозиторий с авторским кодом и нашими файлами |
| `F:\AIRI_task\.venv` | отдельная среда Linux Python 3.11.16 |
| `F:\AIRI_task\.pythons` | управляемый Python, без изменения Windows Miniconda |
| `F:\AIRI_task\.tools` | локальный `uv` 0.12.21 |
| `F:\AIRI_task\.cache` | временный кэш установки и Hugging Face |
| `data/checkpoints/MolmoMotion-4B-H3-F30` | единственный checkpoint для первого запуска |
| `examples/data/davis_bmx_trees` | авторский пример с кадрами и входными точками |
| `runs/` | локальные результаты и smoke tests, исключены из Git |

Среда запуска — Ubuntu 24.04 в WSL2. Установленная в Windows Miniconda на
`F:\Apps\miniconda3` и её окружения не изменялись. В файле
`C:\Users\lesha\.wslconfig` установлено:

```ini
[wsl2]
networkingMode=mirrored
memory=24GB
```

Это восстановило сетевой доступ WSL и повысило его лимит памяти с 16 до 24 ГБ.
Для применения изменений WSL был перезапущен командой `wsl --shutdown`.
В `src/molmo_motion/train/checkpointer.py` сделана одна локальная поправка:
`torch.load(..., mmap=True)` читает большой checkpoint через отображение файла,
снижая пик RAM при загрузке. Численное содержимое весов не меняется.
`scripts/smoke_inference.py` строит модель в BF16 до загрузки весов, то есть
использует ту же итоговую точность, что авторский quickstart.

## Быстрый старт из нового PowerShell

```powershell
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion
```

В открывшейся Ubuntu:

```bash
source /mnt/f/AIRI_task/.venv/bin/activate
export HF_HOME=/mnt/f/AIRI_task/.cache/huggingface
python scripts/check_setup.py
```

Проверка должна показать `torch_cuda_runtime: 12.8`, GPU RTX 4070,
`molmo_motion_imports` и число кадров тестового видео. Самостоятельно проверить
GPU можно командой `nvidia-smi`. `torch.cuda.is_available()` и небольшая
CUDA-операция входят в `check_setup.py`.

Проверить авторский путь визуализации без запуска модели:

```bash
python examples/01_quickstart.py --from-prediction \
  --output runs/preview_from_bundled_prediction.mp4
```

Этот MP4 строится из *готового* прогноза авторов и не доказывает, что модель
была запущена локально.

Проверенный локальный smoke test загружает модель и предсказывает **один**
будущий кадр для восьми точек встроенного эпизода:

```bash
python scripts/smoke_inference.py
```

Результаты сохраняются в `runs/smoke_real_f1.json` и
`runs/smoke_real_f1.npz`. На этой машине прогон прошёл за 68,35 с после
импорта Python, достигнув 9,86 GiB зарезервированной VRAM и 22,32 GiB
пикового RSS процесса. Это подтверждает запуск модели, но ещё не является
оценкой качества на полном горизонте F=30.

## Восстановление среды, если она отсутствует

В Ubuntu из корня репозитория:

```bash
mkdir -p /mnt/f/AIRI_task/.tools /mnt/f/AIRI_task/.cache
curl -LsSf https://astral.sh/uv/0.12.21/install.sh \
  -o /mnt/f/AIRI_task/.tools/uv-install.sh
UV_INSTALL_DIR=/mnt/f/AIRI_task/.tools UV_NO_MODIFY_PATH=1 \
  sh /mnt/f/AIRI_task/.tools/uv-install.sh

export UV_CACHE_DIR=/mnt/f/AIRI_task/.cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/f/AIRI_task/.pythons
UV=/mnt/f/AIRI_task/.tools/uv
"$UV" python install 3.11.16
"$UV" venv --python 3.11.16 /mnt/f/AIRI_task/.venv
"$UV" pip install --python /mnt/f/AIRI_task/.venv/bin/python \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python /mnt/f/AIRI_task/.venv/bin/python \
  torchcodec==0.9.1 --index-url https://download.pytorch.org/whl/cpu --no-deps
"$UV" pip install --python /mnt/f/AIRI_task/.venv/bin/python -e '.[viz]'
"$UV" pip install --python /mnt/f/AIRI_task/.venv/bin/python pip==25.2
```

На этом компьютере бинарный `uv` находится в
`.tools/uv-x86_64-unknown-linux-gnu/uv`, поскольку он был распакован вручную,
когда сеть WSL ещё не работала. Обе формы — один и тот же `uv` 0.12.21.
Полный снимок установленных версий: `configs/installed-packages-wsl.txt`.
Изолированная среда использует PyTorch 2.9.1+cu128, torchvision
0.24.1+cu128, CPU-декодер torchcodec 0.9.1 и Transformers 4.57.6. Для
импорта TorchCodec нужны общие библиотеки FFmpeg 6.1 из Ubuntu:

```bash
sudo apt-get install -y --no-install-recommends \
  libavdevice60 libavfilter9 libswscale7 libswresample4
```

CUDA-wheel TorchCodec 0.9.1 требует ещё NVIDIA NPP, а для этого проекта
видео декодируется на CPU. CPU-wheel устраняет лишнюю зависимость CUDA
декодера и сохраняет совместимость с PyTorch 2.9.

## Веса и входные данные

Для авторского `MolmoMotion.from_pretrained` нужны `config.yaml` и
`model.pt` из `allenai/MolmoMotion-4B-H3-F30`. В репозитории Hugging Face также
есть четыре `safetensors`-шарда для другого способа загрузки. Они здесь не
скачивались, чтобы не хранить второй набор весов. Зафиксирован revision
`3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`.

```bash
source /mnt/f/AIRI_task/.venv/bin/activate
export HF_HOME=/mnt/f/AIRI_task/.cache/huggingface
hf download allenai/MolmoMotion-4B-H3-F30 config.yaml model.pt \
  --revision 3f5e790a511ff2cdf21c8d2a14cb4d8409c94629 \
  --local-dir data/checkpoints/MolmoMotion-4B-H3-F30
```

`model.pt` занимает 17 847 894 425 байт; перед повторной загрузкой проверьте
свободное место. Для первой проверки достаточно встроенного примера
`davis_bmx_trees`; MolmoMotion-1M и полный PointMotionBench не требуются.
При настоящем запуске передавайте **локальный путь** checkpoint: вызов с
ID репозитория Hugging Face внутри авторского загрузчика использует
`snapshot_download` и может скачать оба формата весов.

## Что проверено и что запускать дальше

Подробные факты и ограничения находятся в `SETUP_DIAGNOSTICS.md`. Первый
содержательный эксперимент после подготовки — запустить настоящий прогноз на
встроенном примере и сопоставить его с реальным продолжением; затем переходить
к подготовке одного эпизода ShareRobot. Веса, датасеты и результаты исключены
из Git правилами `.gitignore`.

Если `pip check` сообщает `decord 0.6.0 is not supported on this platform`,
причина — тег `cp36-cp36m` в старом wheel. В этом окружении Decord 0.6.0
импортирован и прочитал видео (122 кадра); это оставшаяся проблема метаданных
пакета, а не ошибка декодирования.
