# Данные первого эксперимента AIRI

В репозитории сохранены исходный код, документ задания, входы эпизода, 33 кадра нужной последовательности, небольшие файлы 2D/3D-разметки, исходный ответ модели, прогноз, эталон, метрики и визуализации. Полная история диалога и локальное Python-окружение не относятся к публикуемому проекту.

## Источники и атрибуция

- Модель и код: [Ai2 MolmoMotion](https://github.com/allenai/molmo-motion), исходный коммит `61f5b21b694ad8f854ec7ecd2400005acc73f685`; лицензия кода в [LICENSE](LICENSE). Checkpoint `allenai/MolmoMotion-4B-H3-F30`, ревизия `3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`, доступен [на Hugging Face](https://huggingface.co/allenai/MolmoMotion-4B-H3-F30).
- Эталонные треки `bmx-trees_2d.npz` и `bmx-trees_3d.npz`: [Ai2 PointMotionBench](https://huggingface.co/datasets/allenai/PointMotionBench), ревизия `564ffa2e3cdb0ba443db8590b40e9691f487436c`. Их размер — 121 и 160 КБ. Правила использования описаны в карточке датасета.
- RGB кадры `bmx-trees`: [DAVIS 2017 TrainVal 480p](https://davischallenge.org/davis2017/code.html), источник файла `DAVIS-2017-trainval-480p.zip`. Атрибуция: Jordi Pont-Tuset и соавторы, [The 2017 DAVIS Challenge on Video Object Segmentation](https://arxiv.org/abs/1704.00675). Проект DAVIS указывает [CC BY-NC 4.0](https://davischallenge.org/) для набора данных. Опубликованы только 33 кадра одного исследуемого эпизода. Рисунки и видео изменены добавлением точек, траекторий и подписей; файлы предназначены для исследовательского некоммерческого использования.

## Большие локальные файлы

GitHub ограничивает обычный Git [100 МБ на файл](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); даже Git LFS ограничен [5 ГБ на файл](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). Поэтому два скачанных исходных файла остались локально:

| Файл | Размер | Как восстановить |
|---|---:|---|
| `data/checkpoints/MolmoMotion-4B-H3-F30/model.pt` | 17 847 894 425 байт | Команда `hf download` в [README_SETUP.md](README_SETUP.md), зафиксированная ревизия выше |
| `data/pointmotionbench/davis/DAVIS-2017-trainval-480p.zip` | 832 766 765 байт | Команда `curl.exe` в [отчёте](report/author_davis.md) и официальный источник DAVIS |

SHA-256 архива DAVIS, двух файлов разметки, скриптов и сохранённого прогноза записаны в [provenance.json](runs/author_davis_bmx_trees_f30/provenance.json); хеш конфигурации и ревизия checkpoint — в [run_status.json](runs/author_davis_bmx_trees_f30/run_status.json). Сохранённые `prediction.npz`, `ground_truth.npz` и результаты оценки можно использовать без скачивания больших файлов и без GPU. Для повторного запуска модели необходим checkpoint.
