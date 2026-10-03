# Полный комплект оставшихся локальных данных

В `main` добавлены ещё 4614 файлов (974 030 371 байт): исходные кадры и depth FMB, RGB-D и COLMAP базы Dobb·E, полные логи, промежуточные результаты, PLEX, материалы DAVIS и WorldTrack. Контрольные суммы всех этих файлов проверены по точным байтам Git index до коммита.

Семь файлов больше лимита обычного Git публикуются в [GitHub Releases этого репозитория](https://github.com/AlexeyPetrov1/Airi_research_task/releases/tag/research-local-assets-2026-10-03). Это оба checkpoint MolmoMotion, полный архив DAVIS, три больших исходных файла FMB и собранное расширение ViPE. Каждый checkpoint разбит на последовательные части по 1 GiB; исходные веса сохраняются без преобразования.

**Загрузка Releases продолжается.** В [описи](remaining_assets_manifest.json) поле `complete` каждого файла и `release_upload_status` показывают подтверждённое состояние. Там же находятся исходные пути, размеры, SHA-256 и ссылки на уже проверенные части. Скрипт отказывается восстанавливать файл, пока загружены не все его части.

После клонирования репозитория для восстановления больших файлов:

```powershell
python scripts/restore_remaining_assets.py
```

Для одного файла:

```powershell
python scripts/restore_remaining_assets.py --only data/fmb_second_scene/raw/source_demo.npy
```

Скрипт проверяет SHA-256 каждой части и собранного файла. Существующий файл с другой контрольной суммой не перезаписывается. Проверка уже восстановленных файлов без загрузки:

```powershell
python scripts/restore_remaining_assets.py --verify-only
```

Две служебные символические ссылки старого F2-NeRF эксперимента описаны в поле `symlinks` описи. В Linux их можно восстановить относительно репозитория:

```bash
ln -s ../second/rgb_prefix runs/dobbe_rgbd_study/f2nerf_second/images
ln -s ../second/colmap_pinhole_all_fixedpp_init62_107/sparse runs/dobbe_rgbd_study/f2nerf_second/sparse
```

Кеши Python/pytest/Hugging Face и автоматически сгенерированные метаданные установленного пакета не входят в исследовательский комплект. Их пути перечислены в `excluded`. Публикация файлов сохраняет исходные ограничения и выводы экспериментов.
