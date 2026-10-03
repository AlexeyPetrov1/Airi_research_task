"""Build final readable indexes and secondary tables from saved real evidence."""
import json
from berkeley_temporal_diagnostics import OUT


def read(path):return json.loads((OUT/path).read_text(encoding='utf8'))
def write(path,value):(OUT/path).write_text(value,encoding='utf8')


def main():
    review=read('visual_review_final.json')
    if review['status']!='complete':raise ValueError('Finish actual visual inspection first')
    fixed=read('fixed_summary.json');oracle=read('oracle_summary.json');rows=[];controls=[]
    for scene in ('cup','bottle'):
        for method,metrics in fixed[scene].items():
            rows.append(dict(scene=scene,method='BASELINE-U/'+method,points=24,metrics=metrics,source='fixed_summary.json'))
        for case in ('CASE-AUGE','CASE-NOK','CASE-NOK-STRICT'):
            path=f'{case}/{scene}/comparison.json';result=read(path)
            if result['point_count']!=24 or result['visual_groups_saved']!=[0,1,2]:raise ValueError('Missing full group results')
            for method,metrics in result['methods'].items():
                entry=dict(scene=scene,method=method,points=24,metrics=metrics,source=path)
                if method.startswith(case):rows.append(entry)
                elif method in ('static','constant_velocity'):controls.append(dict(geometry=case,**entry))
    h1=read('CASE-H1/cup/comparison.json')
    result=dict(status='local_research_complete',evaluation_priority='visual',
        genuine_new_MolmoMotion_forward_calls=19,genuine_MoGe_observed_RGB_frames=16,
        full_24_ID_cases=6,frozen_scenes=['cup episode10 t0=63','bottle episode9 t0=47'],
        new_grounding_segmentation_tracking_calls=0,
        full_cases_secondary=rows,geometry_specific_controls_secondary=controls,
        future_fitted_oracle_diagnostic=oracle,H1_first_8_control=h1,
        visual_review='visual_review_final.json',share_mapping='sources/sharerobot_pixel_mapping.json',
        verification='final_verification.json',tests='tests_receipt.json',
        limitation='Improvement relative to original MolmoMotion does not establish superiority over CV or a pure FPS cause. Kalib calibration not run without verified TCP correspondences.')
    write('summary.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Дополнительные численные результаты','',
        'Главные выводы сформулированы по изображениям в [полном рассказе](research_story.md) и [визуальном отчёте](visual_review_final.json). Ниже все 24 неизменных ID; строки H1 и oracle отделены по размеру выборки и использованию будущего. Значения 3D_est зависят от оценённой K и приведены только в JSON.','',
        '| Сцена | Вариант | ADE 2D, px | FDE 2D, px |','|---|---|---:|---:|']
    for row in rows:
        m=row['metrics']['2D_px'];lines.append(f"| {row['scene']} | {row['method']} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} |")
    lines+=['','Контроли вычислены в геометрии соответствующей ветки, из её H3, на том же 2D reference:','',
        '| Сцена | Геометрия | Контроль | ADE 2D, px | FDE 2D, px |','|---|---|---|---:|---:|']
    for row in controls:
        m=row['metrics']['2D_px'];lines.append(f"| {row['scene']} | {row['geometry']} | {row['method']} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} |")
    lines+=['','Oracle: коэффициент подбирается по будущей 3D_est траектории. Это диагностика, не результат независимого предсказания.','',
        '| Сцена | Вариант | ADE 2D, px | FDE 2D, px |','|---|---|---:|---:|']
    for scene,methods in oracle['scenes'].items():
        for method,metrics in methods.items():
            m=metrics['2D_px'];lines.append(f"| {scene} | {method} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} |")
    lines+=['','H1: только первые восемь точек чашки, полный raw ответ содержит 32 шага; оценка на тех же десяти физических моментах.','',
        '| Вариант | ADE 2D, px | FDE 2D, px |','|---|---:|---:|']
    for method,metrics in h1['methods'].items():
        m=metrics['2D_px'];lines.append(f"| {method} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} |")
    lines+=['','[Полные числа и источники каждой строки](summary.json).']
    write('summary.md','\n'.join(lines)+'\n')
    write('README.md','''# Улучшение Berkeley UR5 → MolmoMotion

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
'''+ '\n'.join(f'| {case} | {scene} | [MP4]({case}/{scene}/geometry_comparison.mp4), [PNG]({case}/{scene}/geometry_contact_sheet.png) | [MP4]({case}/{scene}/geometry_group_01_comparison.mp4), [PNG]({case}/{scene}/geometry_group_01_contact_sheet.png) | [MP4]({case}/{scene}/geometry_group_02_comparison.mp4), [PNG]({case}/{scene}/geometry_group_02_contact_sheet.png) |' for case in ('CASE-AUGE','CASE-NOK','CASE-NOK-STRICT') for scene in ('cup','bottle'))+'''

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
''')
    write('work_plan.md','''# Итог выполнения цели

Основа — неизменный baseline `4f11c56545400c437f822ad677c48a3dd2655347`; [полное описание](request.txt). Визуальная оценка предшествовала решению расширить настоящие P8-пилоты на 24 точки. Все самостоятельные исследования из описания обработаны.

| Пункт | Итог |
|---|---|
| Собственное перемещение ×1/3; независимый step1→frame1 | Выполнены на 24 ID обеих сцен, массивы/PNG/MP4 сохранены. Размах меньше, ошибки направления/фазы остаются. |
| Oracle alpha и cup↔bottle перенос | Выполнены; future-fitted диагностика отделена от независимого прогноза. |
| Raw/smoothed H3 скорость | Выполнено; систематического ×3 нет. |
| CASE-AUGE | Официальный вертикальный FOV подтверждён; настоящий пилот, затем 24 точки обеих сцен. Чашка изменилась, бутылка сохраняет дугу. |
| CASE-NOK | DELTA default f=W проверен и признан недостаточным для строгого условия. Настоящий MoGe-2 fov_x=None; лучевой контроль на 24 точках обеих сцен. |
| CASE-NOK-STRICT | Новые lift/trust/anchors/filter/smoothing без старых K/XYZ/trust; настоящий пилот, затем 24 точки обеих сцен. ID 30 бутылки сохранён и отдельно просмотрен. |
| H1-F32 cup | Настоящий контроль первых 8 ID, все 32 шага сохранены. Раннее размещение остаётся. |
| ShareRobot↔Berkeley | Закрыто точным совпадением 60 RGB с уникальными native source frame indices и возрастающим порядком времени. |
| Kalib, дополнительная проверка | Официальный код и требования исследованы. Калибровка не выполнена без известных TCP↔pixel соответствий; объектные точки не подменяют TCP. |
| Проверки, рассказ, сохранение | Все 18 полных P8-групп и H1 просмотрены, all24 графики сохранены, 10 тестов пройдены; итоговый аудит и публикация выполняются завершающим шагом. |

MolmoPoint, официальная SAM 2.1, K-means и AllTracker не перезапускались. Сцены, t0, 24 ID, RGB, маски, 2D tracks, native depth, instruction и future tracker reference сохранены. Геометрия новых входов использует только observed данные. Oracle использует будущее явно и только для диагностики. AugE, MoGe и DELTA default не названы измеренной калибровкой.

[Полный рассказ](research_story.md) · [Финальная визуальная оценка](visual_review_final.json) · [Финальный аудит](final_verification.json) · [Индекс результатов](README.md).
''')
    story=(OUT/'research_story.md').read_text(encoding='utf8')
    story=story.replace('На момент этой записи завершены временные диагностики, полные CASE-AUGE и CASE-NOK, контроль H1 и точное сопоставление ShareRobot. Строгая ветка CASE-NOK-STRICT завершила чашку; полный результат бутылки ещё рассчитывается. Окончательный статус и сводные результаты будут зафиксированы после его проверки.',
        'Все исследования этого этапа завершены: временные диагностики, полные CASE-AUGE, CASE-NOK и CASE-NOK-STRICT на 24 точках обеих сцен, H1-контроль и точное сопоставление ShareRobot. По Kalib выполнен аудит применимости; сама калибровка требует отсутствующих проверенных TCP соответствий. [Индекс визуальных результатов](README.md), [финальная визуальная оценка](visual_review_final.json) и [итоговые числа](summary.md) сохранены рядом.')
    story=story.replace('Визуальная оценка фиксирует, какие кадры и группы действительно просмотрены, в [visual_review.json](visual_review.json).',
        'Визуальная оценка фиксирует, какие кадры и группы действительно просмотрены, в [visual_review_final.json](visual_review_final.json). Во всех 18 группах полных веток просмотрены три момента всех колонок и все десять кадров скорректированной колонки; непрерывное воспроизведение видео не заявляется. Полные графики всех 24 точек показывают также выход за границы изображения.')
    marker='## Что дал H1 и почему смотрим все группы'
    if 'Полный строгий результат подтвердил' not in story:
        story=story.replace(marker,'''Полный строгий результат подтвердил изменение формы движения чашки во всех группах: преобладает подъём. После `1/3` точки гораздо ближе к чашке, но начало подъёма слишком быстрое, а третья группа к концу несколько ниже объекта. Строгая ветка не стала однозначно лучше лучевого контроля на всех моментах. У бутылки большая дуга сохранилась и после нового trust/smoothing. Уменьшение размаха не убирает раннее размещение вправо.

Отмеченный ID 30 рассмотрен [отдельно](bottle/strict_retained_id30.png). Его sensor Z в H3 меняется на 12 мм и затем почти не меняется; сглаживание сохраняет лучи изображения, но меняет XYZ максимум на 8,20 мм. Прогноз этой точки тоже преждевременно уходит вправо. Её не исключили; просмотр не отменяет ограничение доверия к глубине. Этот факт важен для честного сравнения общего набора.

'''+marker)
    marker='## Kalib и практическая воспроизводимость'
    if 'round(linspace' not in story:
        story=story.replace(marker,'''Ещё одна находка: точные исходные индексы обеих последовательностей совпадают с `round(linspace(0,N−1,31))[:-1]`, где N=120 для чашки и N=97 для бутылки. Обычные 30 равноотстоящих индексов с включённым последним кадром не совпадают. Это восстановление правила по проверенным пикселям, а не найденная реализация автора. [Ответ автора](https://github.com/FlagOpen/ShareRobot/issues/4#issuecomment-3094378733) подтверждает сохранение исходных episode ID и равномерную выборку 30 кадров, но сам по себе не заменяет наше доказательство изображений.

'''+marker)
    if 'В сумме сделано 19' not in story:
        story+='\nВ сумме сделано 19 новых настоящих forward-вызовов MolmoMotion: 6 AugE, 6 лучевого MoGe, 6 строгой геометрии и 1 H1. MoGe отдельно обработал 16 наблюдаемых RGB. Три полные ветки × две сцены дали 18 ответов по 8×30 point-time, H1 — 8×32; ни один ответ не дополнен искусственно. Все 23 сравнивающих видео декодированы в десять кадров при 5 FPS. [Десять тестов](tests_receipt.json) проверяют вычисление собственного смещения, строгий разбор tracks и отбраковку повреждённых PNG. [Финальный аудит](final_verification.json) дополнительно проверяет неизменность исходного baseline и sealed входов новых веток, reconstruction XYZ и все 60 извлечённых PNG.\n'
    write('research_story.md',story)
    headers=read('sources/sharerobot_archive_headers.json')
    headers['limitation']='The author reply supports preservation of source episode IDs and even extraction of 30 frames across datasets. Berkeley pixel identity is independently established by sharerobot_pixel_mapping.json; the reply alone is not image proof.'
    write('sources/sharerobot_archive_headers.json',json.dumps(headers,ensure_ascii=False,indent=2)+'\n')
    manifest=read('all24_visual_manifest.json')
    for item in manifest:item['visual_review_pending']=False;item['review_receipt']='visual_review_final.json'
    write('all24_visual_manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(secondary_rows=len(rows),control_rows=len(controls),full_cases=6)),flush=True)


if __name__=='__main__':main()
