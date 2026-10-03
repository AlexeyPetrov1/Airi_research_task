# Улучшение Berkeley UR5 → MolmoMotion

Исследование по [описанию цели](request.txt) завершено. Сохранены 19 новых настоящих вызовов MolmoMotion: AugE, MoGe с прежним Z и строгая MoGe-геометрия на всех 24 точках обеих сцен, а также H1 на первых восьми точках чашки. Исходный [baseline](../berkeley_ur5_molmomotion/) не изменён; настоящая официальная SAM 2.1, MolmoPoint и AllTracker переиспользованы.

**Главный визуальный результат:** у чашки MoGe и строгая ветка заменяют широкую дугу размещения согласованным подъёмом всех трёх групп. После фиксированного уменьшения собственного перемещения каждой точки в три раза прогноз гораздо ближе к чашке. Ранняя скорость остаётся завышенной. У бутылки широкая дуга и ранний уход вправо сохраняются. Единственная причина в виде FPS не установлена; линейная экстраполяция остаётся сильнее на этих reference.

[Полный понятный рассказ о процессе, находках и ограничениях](research_story.md) · [План и итог каждого пункта](work_plan.md) · [Визуальная оценка](visual_review_final.json) · [Дополнительные числа](summary.md).

## Сначала посмотрите траектории

На общих графиках показаны все 24 ID и все 30 шагов модели; зелёный — tracker reference, синий/оранжевый/розовый — три независимых P8-группы. Масштабы осей общие для вариантов, выход за изображение не обрезан.

| Сцена | Исходные выходы всех веток | Собственное перемещение ×1/3 | Фиксированный контроль и изменение времени |
|---|---|---|---|
| Чашка | [Все 24 точки](cup/all24_actual.png) | [Все 24 точки](cup/all24_one_third.png) | [PNG](cup/fixed_contact_sheet.png), [MP4](cup/fixed_comparison.mp4) |
| Бутылка | [Все 24 точки](bottle/all24_actual.png) | [Все 24 точки](bottle/all24_one_third.png) | [PNG](bottle/fixed_contact_sheet.png), [MP4](bottle/fixed_comparison.mp4) |

Ниже каждое видео показывает одну группу на одинаковых настоящих будущих RGB, при 5 FPS: baseline / новый raw прогноз / новый прогноз ×1/3. Зелёные точки и линии — будущий tracker reference, розовые — прогноз. Contact sheets показывают +0,2, +1,0 и +2,0 с. В каждой папке есть также PNG всех десяти кадров и полосы просмотренной скорректированной колонки в `video_audit/`.

| Ветка | Сцена | Группа 00 | Группа 01 | Группа 02 |
|---|---|---|---|---|
| CASE-AUGE | cup | [MP4](CASE-AUGE/cup/geometry_comparison.mp4), [PNG](CASE-AUGE/cup/geometry_contact_sheet.png) | [MP4](CASE-AUGE/cup/geometry_group_01_comparison.mp4), [PNG](CASE-AUGE/cup/geometry_group_01_contact_sheet.png) | [MP4](CASE-AUGE/cup/geometry_group_02_comparison.mp4), [PNG](CASE-AUGE/cup/geometry_group_02_contact_sheet.png) |
| CASE-AUGE | bottle | [MP4](CASE-AUGE/bottle/geometry_comparison.mp4), [PNG](CASE-AUGE/bottle/geometry_contact_sheet.png) | [MP4](CASE-AUGE/bottle/geometry_group_01_comparison.mp4), [PNG](CASE-AUGE/bottle/geometry_group_01_contact_sheet.png) | [MP4](CASE-AUGE/bottle/geometry_group_02_comparison.mp4), [PNG](CASE-AUGE/bottle/geometry_group_02_contact_sheet.png) |
| CASE-NOK | cup | [MP4](CASE-NOK/cup/geometry_comparison.mp4), [PNG](CASE-NOK/cup/geometry_contact_sheet.png) | [MP4](CASE-NOK/cup/geometry_group_01_comparison.mp4), [PNG](CASE-NOK/cup/geometry_group_01_contact_sheet.png) | [MP4](CASE-NOK/cup/geometry_group_02_comparison.mp4), [PNG](CASE-NOK/cup/geometry_group_02_contact_sheet.png) |
| CASE-NOK | bottle | [MP4](CASE-NOK/bottle/geometry_comparison.mp4), [PNG](CASE-NOK/bottle/geometry_contact_sheet.png) | [MP4](CASE-NOK/bottle/geometry_group_01_comparison.mp4), [PNG](CASE-NOK/bottle/geometry_group_01_contact_sheet.png) | [MP4](CASE-NOK/bottle/geometry_group_02_comparison.mp4), [PNG](CASE-NOK/bottle/geometry_group_02_contact_sheet.png) |
| CASE-NOK-STRICT | cup | [MP4](CASE-NOK-STRICT/cup/geometry_comparison.mp4), [PNG](CASE-NOK-STRICT/cup/geometry_contact_sheet.png) | [MP4](CASE-NOK-STRICT/cup/geometry_group_01_comparison.mp4), [PNG](CASE-NOK-STRICT/cup/geometry_group_01_contact_sheet.png) | [MP4](CASE-NOK-STRICT/cup/geometry_group_02_comparison.mp4), [PNG](CASE-NOK-STRICT/cup/geometry_group_02_contact_sheet.png) |
| CASE-NOK-STRICT | bottle | [MP4](CASE-NOK-STRICT/bottle/geometry_comparison.mp4), [PNG](CASE-NOK-STRICT/bottle/geometry_contact_sheet.png) | [MP4](CASE-NOK-STRICT/bottle/geometry_group_01_comparison.mp4), [PNG](CASE-NOK-STRICT/bottle/geometry_group_01_contact_sheet.png) | [MP4](CASE-NOK-STRICT/bottle/geometry_group_02_comparison.mp4), [PNG](CASE-NOK-STRICT/bottle/geometry_group_02_contact_sheet.png) |

[H1-контроль чашки, 8 точек](CASE-H1/cup/geometry_contact_sheet.png) · [Видео H1](CASE-H1/cup/geometry_comparison.mp4) · [Oracle чашки](cup/oracle_contact_sheet.png) · [Oracle бутылки](bottle/oracle_contact_sheet.png) · [Отмеченный, сохранённый ID 30 бутылки](bottle/strict_retained_id30.png).

## Что установлено

Фиксированное `1/3` уменьшает избыточный размах, но не всегда исправляет форму и фазу. Oracle по всей будущей траектории даёт 0,1657 и 0,1954; совпадение конечной длины не означает совпадения движения. Авторское сглаживание не ускоряет H3 систематически втрое. [Аудит времени модели](sources/model_time_contract.json) обнаружил ordinal timestamps и в processor, и в training builder; отдельная ошибка processor этим не доказана.

AugE использует официальный вертикальный FOV симуляции; это кандидатная камера. MoGe-2 действительно оценил лучи по восьми observed RGB каждой сцены, без K/FOV. CASE-NOK сохраняет старый smoothed sensor Z для контроля лучей; CASE-NOK-STRICT заново считает lift, trust, anchors, filter и smoothing по native depth и MoGe-лучам. [Строгий аудит чашки](CASE-NOK-STRICT/cup/geometry/strict_geometry_audit.json) и [бутылки](CASE-NOK-STRICT/bottle/geometry/strict_geometry_audit.json) подтверждают отсутствие старых K/XYZ/trust в расчёте. Все ID сохранены, включая отмеченный фильтром ID 30.

Связь ShareRobot закрыта **точным совпадением всех 60 RGB** с уникальными кадрами исходного RLDS. [Доказательство и индексы времени](sources/sharerobot_pixel_mapping.json), [пары чашки](sources/share_native_pixel_pairs_cup.png), [пары бутылки](sources/share_native_pixel_pairs_bottle.png). PNG извлечены из нужного участка split gzip без скачивания всего 343-ГБ архива. Редкие planning-кадры охватывают весь эпизод; плотная H3-история осталась исходной.

[Kalib](sources/kalib_readiness.json) проверен как дополнительный метод. Реальная калибровка не выполнена: нужны проверенные TCP 3D↔pixel соответствия, а объектные точки не являются TCP. Его применимость и недостающие данные описаны явно.

## Проверки и воспроизводимость

[Фиксированный протокол и SHA256](protocol.json) · [Итоговый аудит](final_verification.json) · [10 пройденных тестов](tests_receipt.json) · [Числа JSON](summary.json). Аудит проверяет все исходные файлы baseline, frozen входы шести полных веток, все 19 raw ответов и восстановление XYZ, 23 MP4 по десять кадров при 5 FPS, SHA256 и CRC всех 60 PNG. Сохранены payloads, input freezes, GPU receipts и снимки исполняемого inference-скрипта для новых полных запусков. Ограничение происхождения кода ранних пилотов описано в рассказе.

Из корня репозитория в окружении эксперимента:

```bash
python -m pytest -q tests/test_berkeley_temporal_diagnostics.py tests/test_berkeley_geometry_tracks.py tests/test_berkeley_share_ranges.py
python scripts/berkeley_improvement_verify.py --final
```

Для нового запуска используйте отдельный каталог/checkout. Подготовка отказывается перезаписывать завершённые входы. Большие веса и полный архив датасета не входят в результаты; закреплённые revisions, SHA256 и доказательные артефакты сохранены. `progress_verification.json`, первые `visual_review*.json`, очереди и начальные mapping probes отражают исторические этапы; окончательные статусы находятся в перечисленных итоговых файлах.
