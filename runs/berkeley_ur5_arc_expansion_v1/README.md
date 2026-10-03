# Небольшое расширение траекторий Berkeley

Общий осторожный вариант: **после +1 с плавно увеличить α с 1/3 до 0,36 и добавить до 5 мм по −Y камеры**. До +1 с прогноз совпадает с прежним ×1/3. К +2 с он дальше примерно на 10–11 пикселей и выше на 10 пикселей в обеих сценах. Это выбранный вариант для небольшого изменения дуги; улучшения точности на обеих сценах нет.

[Открыть интерактивное сравнение всех 15 вариантов и трёх бейзлайнов](gallery_phase.html). Открывается локально в браузере после клонирования репозитория; GitHub показывает HTML как исходный код. Один переключатель меняет сразу обе сцены. Доступны все 24 точки, три отдельные P8 группы, десять настоящих моментов и воспроизведение 5 FPS.

[Полный понятный разбор процесса и выводов](research_story.md) · [Визуальная оценка](visual_review.json) · [Выбранные массивы и параметры](selected/metadata.json).

| Сцена | Метод | ADE, px ↓ | FDE, px ↓ |
|---|---|---:|---:|
| cup | reference_0333 | 28.111 | 36.227 |
| cup | scale_036_lift005 | 33.597 | 39.234 |
| cup | lift_010 | 30.555 | 36.694 |
| cup | CV_3D | 2.906 | 7.374 |
| cup | object_translation_3D | 1.990 | 5.614 |
| cup | late_036_lift005 | 29.773 | 39.234 |
| bottle | reference_0333 | 44.650 | 34.255 |
| bottle | scale_036_lift005 | 50.625 | 33.578 |
| bottle | lift_010 | 44.966 | 29.660 |
| bottle | CV_3D | 11.045 | 11.934 |
| bottle | object_translation_3D | 11.370 | 10.895 |
| bottle | late_036_lift005 | 45.692 | 33.578 |

K в обеих сценах побитово одинаковый: fx = fy = 434,5584411621094, cx = 320, cy = 240. Точное значение из FOV до сохранения float32 — 434,5584412271572. Используются сохранённые настоящие CASE-AUGE predictions всех трёх групп. Новый inference и обучение не выполнялись; SAM, RGB, history, depth и point IDs не менялись.

Статика, per-point CV и общее медианное перемещение построены только по наблюдавшимся данным. CV остаётся значительно ближе к reference на этих коротких окнах. Лучшее FDE отдельных вариантов не означает лучшую траекторию по всем моментам: одинаковые концы constant/late вариантов дают одинаковый FDE, но разную ошибку в середине.

Сначала заморожены 11 вариантов масштаба/высоты, затем после визуальной диагностики преждевременного движения — четыре одинаковых для обеих сцен временных правила. Это исследовательская проверка уже просмотренных сцен; численное подбирание коэффициента по будущему не применялось. Выбор и гипотеза используют будущие evaluation кадры и не являются независимым test split.

| Сцена | Данные | Все 24 точки | Три группы и реальные кадры |
|---|---|---|---|
| cup | episode 10, t0=63 | [сравнение](selected/cup/all24_comparison.png), [3D NPY](selected/cup/postprocessed_future_3d.npy) | [группа 0 MP4](phase_schedule/cup/timing_group_00.mp4), [группа 1 MP4](phase_schedule/cup/timing_group_01.mp4), [группа 2 MP4](phase_schedule/cup/timing_group_02.mp4) |
| bottle | episode 9, t0=47 | [сравнение](selected/bottle/all24_comparison.png), [3D NPY](selected/bottle/postprocessed_future_3d.npy) | [группа 0 MP4](phase_schedule/bottle/timing_group_00.mp4), [группа 1 MP4](phase_schedule/bottle/timing_group_01.mp4), [группа 2 MP4](phase_schedule/bottle/timing_group_02.mp4) |

Все 30 будущих steps сохранены; оценка берёт indices [2,5,…,29] и реальные моменты +0,2…+2,0 с. Всего 30 MP4: 18 scale/height/control и 12 phase. В каждой папке есть полные XYZ/UV NPZ, contact sheets и монтажи всех десяти декодированных кадров. Для монтажей время идёт четырьмя столбцами; на каждый блок времени три вертикальные строки соответствуют трём вариантам из render_receipts.json. Сами MP4 сохраняют полный исходный кадр.

[Все метрики 11 основных вариантов](results.json), [четырёх временных правил](phase_schedule/results.json), [замороженный основной протокол](protocol.json), [протокол продолжения](phase_schedule/protocol.json), [полная проверка 30 прогнозов и 30 видео](final_verification.json), [ресурсы GitHub/Hugging Face](resource_provenance.json).

Метрики вторичны: native 2D ADE/FDE, PWT<5/10/20/40px, E(t), направление и скорость, длина пути, конечное перемещение, 3D pair-distance drift. 3D_est использует tracker и native Z в гипотезе K и не является calibrated ground truth. Предсказания вне кадра не отбрасываются из оценки.

Для проверки сохранённых результатов:

```bash
python -m pytest tests/test_berkeley_arc_expansion.py -q
python scripts/berkeley_arc_expansion.py --stage verify
python scripts/berkeley_arc_verify.py
```

Для повторения в новой папке (старые runs не перезаписываются):

```bash
python scripts/berkeley_arc_expansion.py --stage generate --output-dir runs/berkeley_arc_repeat
python scripts/berkeley_arc_phase.py --parent-run runs/berkeley_arc_repeat
```

[Предыдущее исследование и источник настоящих модельных прогнозов](../berkeley_ur5_improvement_v1/research_story.md).
