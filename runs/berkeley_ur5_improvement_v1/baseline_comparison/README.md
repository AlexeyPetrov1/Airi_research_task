# Сравнение траекторий с причинными бейзлайнами

Все 24 фиксированных ID, десять настоящих будущих моментов +0,2…+2,0 с, один общий tracker reference. Будущее не использовано для построения бейзлайнов. Выбор лучших существующих веток после просмотра этих сцен — исследовательский выбор, не результат на новом test split.

| Сцена | Метод | ADE px ↓ | FDE px ↓ | PWT@10px ↑ | PWT@20px ↑ | Длина пути / reference |
|---|---|---:|---:|---:|---:|---:|
| cup | MolmoMotion_original | 193.571 | 224.485 | 0.0% | 0.0% | 6.621 |
| cup | MolmoMotion_original_x1_3 | 46.917 | 69.211 | 10.0% | 18.8% | 2.091 |
| cup | step_equals_frame_diagnostic | 54.436 | 116.072 | 16.2% | 23.3% | 2.383 |
| cup | static_BASELINE_U | 46.114 | 83.608 | 10.0% | 20.0% | 0.000 |
| cup | CV_3D_BASELINE_U | 2.906 | 7.374 | 98.8% | 100.0% | 1.062 |
| cup | object_translation_3D_BASELINE_U | 1.993 | 5.621 | 100.0% | 100.0% | 1.064 |
| cup | CV_2D_no_K | 2.946 | 5.313 | 100.0% | 100.0% | 0.962 |
| cup | CASE-AUGE_actual | 152.304 | 174.449 | 0.0% | 0.0% | 3.377 |
| cup | CASE-AUGE_x1_3 | 28.111 | 36.227 | 12.5% | 38.8% | 1.121 |
| cup | CASE-NOK_actual | 131.621 | 132.321 | 0.0% | 0.0% | 2.907 |
| cup | CASE-NOK_x1_3 | 19.004 | 18.670 | 12.1% | 54.6% | 0.965 |
| cup | CASE-NOK-STRICT_actual | 125.927 | 118.177 | 0.0% | 0.0% | 2.862 |
| cup | CASE-NOK-STRICT_x1_3 | 18.699 | 22.277 | 11.7% | 53.8% | 0.952 |
| bottle | MolmoMotion_original | 223.409 | 194.015 | 0.0% | 0.0% | 5.014 |
| bottle | MolmoMotion_original_x1_3 | 47.950 | 36.494 | 0.0% | 14.2% | 1.703 |
| bottle | step_equals_frame_diagnostic | 103.084 | 181.552 | 0.0% | 10.0% | 2.792 |
| bottle | static_BASELINE_U | 60.809 | 102.199 | 0.0% | 10.0% | 0.000 |
| bottle | CV_3D_BASELINE_U | 11.045 | 11.934 | 42.1% | 97.9% | 0.917 |
| bottle | object_translation_3D_BASELINE_U | 11.316 | 10.860 | 30.8% | 100.0% | 0.921 |
| bottle | CV_2D_no_K | 15.462 | 20.206 | 23.8% | 67.9% | 0.790 |
| bottle | CASE-AUGE_actual | 218.392 | 201.396 | 0.0% | 0.0% | 4.464 |
| bottle | CASE-AUGE_x1_3 | 44.650 | 34.255 | 0.0% | 11.7% | 1.523 |
| bottle | CASE-NOK_actual | 208.823 | 222.507 | 0.0% | 0.0% | 4.213 |
| bottle | CASE-NOK_x1_3 | 42.275 | 44.254 | 0.0% | 11.2% | 1.430 |
| bottle | CASE-NOK-STRICT_actual | 214.116 | 228.994 | 0.0% | 0.0% | 4.004 |
| bottle | CASE-NOK-STRICT_x1_3 | 45.617 | 50.344 | 0.0% | 10.0% | 1.358 |

CV_2D_no_K — линейная экстраполяция исходных 2D H3 tracks. CV_3D — линейная экстраполяция XYZ по настоящим timestamps с последующей проекцией. object_translation_3D — одна медианная observed скорость всех точек, что сохраняет исходные попарные 3D расстояния. Статика использует t0. H1 и future-fitted oracle имеют иной контракт и остаются в [предыдущей сводке](../summary.md).

Для строгой MoGe-ветки CV пересчитан по тем же её входам; полные результаты для каждой геометрии в JSON. Нативные PWT пороги и диагностическая метрика длины пути не являются официальным TAP-Vid протоколом.

| Сцена | Контроль в геометрии лучшей визуальной ветки | ADE px ↓ | FDE px ↓ | PWT@20px ↑ |
|---|---|---:|---:|---:|
| cup | CASE-NOK-STRICT/static | 46.114 | 83.608 | 20.0% |
| cup | CASE-NOK-STRICT/CV_3D | 2.735 | 6.476 | 100.0% |
| cup | CASE-NOK-STRICT/object_translation_3D | 1.986 | 5.634 | 100.0% |
| bottle | CASE-NOK/static | 60.809 | 102.199 | 10.0% |
| bottle | CASE-NOK/CV_3D | 11.045 | 11.934 | 97.9% |
| bottle | CASE-NOK/object_translation_3D | 11.370 | 10.895 | 100.0% |

Новые contact sheets и MP4 сопоставляют static / CV в той же геометрии / выбранную модель ×1/3 на настоящих RGB; зелёный — tracker reference. Все три группы сохранены.

- cup: [все 24 точки пяти методов и ошибка по времени](cup/all24_baselines.png).
- cup, группа 00: [PNG](cup/baselines_group_00_contact_sheet.png), [MP4](cup/baselines_group_00_comparison.mp4).
- cup, группа 01: [PNG](cup/baselines_group_01_contact_sheet.png), [MP4](cup/baselines_group_01_comparison.mp4).
- cup, группа 02: [PNG](cup/baselines_group_02_contact_sheet.png), [MP4](cup/baselines_group_02_comparison.mp4).
- bottle: [все 24 точки пяти методов и ошибка по времени](bottle/all24_baselines.png).
- bottle, группа 00: [PNG](bottle/baselines_group_00_contact_sheet.png), [MP4](bottle/baselines_group_00_comparison.mp4).
- bottle, группа 01: [PNG](bottle/baselines_group_01_contact_sheet.png), [MP4](bottle/baselines_group_01_comparison.mp4).
- bottle, группа 02: [PNG](bottle/baselines_group_02_contact_sheet.png), [MP4](bottle/baselines_group_02_comparison.mp4).

[Полные метрики, покрытие, горизонты, группы и 3D согласованность](results.json).
