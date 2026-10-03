"""Write the experiment report from final receipts and manual visual review."""
from pathlib import Path
import json
import numpy as np
from das_prepare_control import write_json,sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_molmomotion/cup/das_robot_cup'

def number(x,places=2):return '—' if x is None else f'{x:.{places}f}'

def main():
    metric=json.loads((OUT/'paired_metrics.json').read_text());kin=json.loads((OUT/'kinematic_projection.json').read_text())
    geo=json.loads((OUT/'observed/geometry_audit.json').read_text());review=json.loads((OUT/'qualitative_review.json').read_text())
    codes={'robot_static':'A: неподвижный робот','robot_coupled':'B: связанный робот','robot_coupled_initial_only':'C: связанный робот, только начальный RGB'}
    lines=[]
    for key,name in codes.items():
        m=metric['variants'][key];robot=m['shared_three_robot_to_requested_control'];cup=m['shared_three_cup_to_control']
        real=m['cup_to_real'];quality=m['image_quality'];resource=json.loads((OUT/key/'resource_usage.json').read_text())
        lines.append(f"| {name} | {number(robot['ADE_px'])} / {number(robot['FDE_px'])} ({robot['visible_pairs']}/{robot['total_pairs']}) | {number(cup['ADE_px'])} / {number(cup['FDE_px'])} ({cup['visible_pairs']}/{cup['total_pairs']}) | {number(real['ADE_px'])} ({real['visible_pairs']}/{real['total_pairs']}) | {number(quality['shared_requested_motion_ROI_MAE'])} | {number(quality['temporal_5Hz_local_warp']['mean_warp_MAE_0_255'])} |")
    pair=[]
    for key in ['robot_static','robot_coupled']:
        m=metric['variants'][key];r=m['common_robot_to_requested_control'];contact=m['common_cup_gripper_relative_motion']
        pair.append(f"| {codes[key]} | {number(r['ADE_px'])} / {number(r['FDE_px'])} ({r['visible_pairs']}/{r['total_pairs']}) | {number(contact['mean_offset_error_px'])} | {contact['valid_times']}/10 |")
    resources=[]
    for key in codes:
        r=json.loads((OUT/key/'resource_usage.json').read_text())
        resources.append(f"| {codes[key]} | {number(r['active_seconds'],1)} | {number(r['peak_cuda_allocated_bytes']/2**30)} | {number(r['peak_cuda_reserved_bytes']/2**30)} | {number(r['peak_process_rss_bytes']/2**30)} |")
    fullquality=[]
    for key in codes:
        q=metric['variants'][key]['image_quality']
        fullquality.append(f"| {codes[key]} | {number(q['full_frame_PSNR_dB'])} | {number(q['full_frame_MAE'])} | {number(q['laplacian_variance'])} | {number(q['temporal_5Hz_local_warp']['mean_valid_fraction'])} |")
    visual=[]
    for key in codes:
        r=review['variants'][key];visual.append(f"- **{codes[key]}:** {r['summary_ru']}")
    text=f'''# Робот и стакан: проверка совместного управления DaS

**Результат:** {review['conclusion_ru']}

Эксперимент выполнен в ветке `codex/das-robot-cup-coupling-20261003-1526`, отдельно от продолжающихся экспериментов на `main`. Это проверка гипотезы на одном замороженном эпизоде и одном seed; статистическая значимость и перенос на другие сцены не установлены.

## Видео и доказательства

- [B: новое видео с движущимся роботом](robot_coupled/generated_molmomotion_seed42.mp4).
- [C: тот же control без постоянного пустого reference и vacancy prior](robot_coupled_initial_only/generated_molmomotion_seed42.mp4).
- [A: парный контроль с неподвижным роботом](robot_static/generated_molmomotion_seed42.mp4).
- [A / B / реальное продолжение, только первые 2 секунды](paired_generated_vs_real_2s.mp4).
- [Все три варианта / real, четыре панели](all_three_vs_real_2s.mp4).
- [A / B, все 6 секунд](static_vs_coupled_full_6s.mp4).
- [Сопоставление в точные общие моменты 0, 1, 2 с](paired_results_exact_times.png).
- [Робот: точки и номера звеньев](observed/robot_tracking_points.png), [траектории точек и углы суставов](robot_point_and_joint_trajectories.png), [SAM2.1 маски](observed/robot_masks_contact_sheet.png), [control и геометрический preview](paired_control_review.png).
- [Кинематические проверки](kinematic_projection.json), [все метрики](paired_metrics.json), [ручной просмотр](qualitative_review.json), [проверка артефактов](verification.json), [SHA256 manifest](artifact_manifest.json), [хеши кода](source_code_manifest.json).

Геометрические preview — детерминированная проекция видимых поверхностей, **не результат DaS**. Основные MP4 сохранены из diffusion pipeline; RGB-кадры результата не ретушировались, не композитились и не заменялись на preview. Видео сопоставления подписаны отдельно. В двухсекундном comparison измеренный real кадр выбран ближайшим по времени и подписан своим фактическим временем; количественная оценка использует точную сетку 5 Гц.

## Проверяемая гипотеза и протокол

Прежний control перемещал только стакан. Теперь робот задаётся точками и плотными поверхностями звеньев; стакан жёстко прикреплён к захвату. Проверяется, превращает ли такое управление отделённое движение предмета в согласованное движение робот–предмет **в результате DaS**, а не только в построенном control.

Исходная сцена прежняя: Berkeley UR5, episode 10, `t0=63`, RGB 640×480, наблюдаемая частота 5 Гц, текст `Pick up the blue cup and put it into the brown cup.` Использован прежний действительный прогноз MolmoMotion `[24,30,3]`; модель повторно не запускалась. Его жёсткое приближение даёт целевой pose стакана. Сохранены родные времена прогноза: 0…2 с, 49 кадров при 8 FPS, после кадра 16 pose удерживается до 6 с. Последний кадр имеет timestamp 6,0 с, длительность MP4-контейнера 6,125 с; двухсекундный comparison содержит 17 кадров и имеет container duration 2,125 с. Никакого растягивания прогноза на 6 с нет.

В исходном TFDS состояние содержит шесть углов суставов, pose TCP, состояние захвата и флаг блокировки; LeRobot сохранил лишь pose и состояние захвата. Это подтверждено [официальным описанием Berkeley UR5](https://sites.google.com/view/berkeley-ur5/home). Из ранее проверенного native record экспортированы **только кадры 56…63**, см. [receipt](observed/robot_state_receipt.json). Измеренные будущие углы, будущие RGB/depth, будущие траектории и желаемое реальное конечное положение не использованы при построении движения.

### Точки → суставы → плотный control

1. На наблюдаемом `t0` вручную поставлены точки на роботе, как запросил пользователь. SAM2.1 Hiera Large выделил предплечье, запястье, захват и верхнее звено. Перекрытия масок имеют одного владельца; запястье разделено на два подвижных звена. Это manual grounding, а не новый прогноз MolmoMotion по роботным точкам.
2. Использована номинальная шестисуставная кинематика UR5 с [DH-параметрами производителя](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/). Совместимость с наблюдаемыми состояниями проверена по постоянному flange→TCP offset: максимум ошибки TCP в истории {max(geo['observed_history_TCP_validation_position_error_m'])*1000:.3f} мм. Это проверка короткой наблюдаемой истории, не внешняя калибровка.
3. Camera→base приближённо зарегистрирована по четырём размеченным суставам в `t0`, estimated `K` и measured depth поверхности. Reprojection RMS по этим точкам {np.sqrt(np.mean(np.array(geo['fit_joint_reprojection_error_px'])**2)):.2f} px, максимум {max(geo['fit_joint_reprojection_error_px']):.2f} px. Производственная калибровка UR5 и настоящие extrinsics камеры отсутствуют; положение центра оси за поверхностью оценено приближённо. См. [geometry audit](observed/geometry_audit.json).
4. Для каждого будущего времени решён IK к требуемому положению и ориентации захвата. Суставы ограничены положением и скоростью 2 рад/с; номинальный предел UR5 180°/с указан в [официальном joint limits файле](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/rolling/config/ur5/joint_limits.yaml). Ускорение описано в receipt, не объявлено проверенным аппаратным пределом. Полная CAD-проверка самоколлизий и моментов не выполнялась; траектория предназначена для видеоэксперимента.
5. Полученные углы через FK двигают каждое звено отдельно. Стакан наследует фактически достигнутый flange pose. Ошибка связи в local frame {kin['max_cup_flange_local_attachment_error_m']:.2e} м; это инвариант построения control. Изменение прежнего cup control после ограничения IK: среднее {kin['cup_to_frozen_rigid_forecast_mean_3d_error_m']*1000:.3f} мм / {kin['cup_to_frozen_rigid_forecast_mean_2d_error_px']:.4f} px на 0…2 с. Значимого исправления ошибки MolmoMotion этим не получено.
6. Восстановлено {kin['robot_points']} роботных material points и 3311 точек стакана; цвета каждой физической точки постоянны. Из-за отражающих металлических поверхностей {kin['estimated_robot_depth_points']} роботных depth значений ({kin['estimated_robot_depth_points']/kin['robot_points']*100:.1f}%) оценены ближайшими валидными измерениями внутри звена. Это явно **estimated depth**, а не новые измерения. Для трекинга результата сохранены 14 роботных и прежние 24 cup точки.
7. Статический фон остаётся на месте; из него удалены исходные робот и стакан. Occluded depth восстановлена из соседних наблюдаемых значений. Для A/B использован синтетический пустой RGB reference по `t0`, созданный встроенным imagegen и применённый только внутри масок; [точный prompt и роль](observed/imagegen_receipt.json), [изображение](observed/imagegen_empty_robot_plate.png), [аудит](observed/clean_reference_audit.png). Вне области редактирования RGB совпадает с `t0` побайтно. Real future не использован.

Начальная геометрия точно совпадает с наблюдаемой; форма каждого звена и стакана сохраняется; минимальный зазор видимой cup поверхности над оценённой плоскостью стола {min(kin['visible_cup_min_table_clearance_m'])*1000:.2f} мм. Оценка касается видимой поверхности и приближённой плоскости, а не полной collision geometry. Объект временно выходит из кадра, как в прежнем прогнозе; исчезновение в эти моменты отдельно от деградации идентичности в видимые моменты. Поверхности, которых камера не видела в `t0`, не восстановлены CAD-моделью и оставляют отверстия в dense rendering; они ограничивают качество управления.

### Что одинаково и что меняется

| Условие | A: robot_static | B: robot_coupled | C: robot_coupled_initial_only |
|---|---|---|---|
| Движение стакана | Одно и то же IK-приближение Molmo | То же | То же |
| Робот | Неподвижный | Суставы IK/FK | Тот же control, SHA256 совпадает с B |
| Начальный RGB + CLIP | Наблюдаемый `t0` | Тот же | Тот же |
| Постоянный spatial reference | Пустая синтетическая сцена | Та же | Отключён |
| Vacancy prior | strength 1, context 24 px | Тот же алгоритм | Отключён |
| Модель / seed / шаги | 1.3B-Control / 42 / 25 | Те же | Те же |

**A/B** — парная проверка изменения роботного control при одинаковом cup control. Foreground protection mask вычисляется из соответствующего control тем же алгоритмом; движение робота закономерно меняет и эту маску. Поэтому вывод относится к control **вместе с его одинаково устроенным conditioning pipeline**, а не к изолированному causal эффекту одной карты при фиксированной маске. **C** — дополнительный опыт, меняющий два условия reference/prior после обнаруженной полупрозрачности B. Это не второй чистый A/B тест робота и не причинное доказательство виновности только vacancy prior.

Использована [официальная ветка DaS Wanfun](https://github.com/IGL-HKUST/DiffusionAsShader/tree/47dcaf13f78fe3cafd2a4aed20df1e059917599a), commit `47dcaf13f78fe3cafd2a4aed20df1e059917599a`, с уже обученным [Alibaba PAI Wan2.1-Fun-V1.1-1.3B-Control](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control/blob/a116c52b08182f7a5916eae7a47fcd61a9467a04/README_en.md), revision `a116c52b08182f7a5916eae7a47fcd61a9467a04`. Это обученная control-модель, интегрированная авторами DaS, а не доказанный отдельный роботный DaS fine-tune. Модельная карточка заявляет trajectory control; это не гарантия сохранения robotic contact и cup identity. Vacancy prior — наш дополнительный inference adapter, а не штатный алгоритм DaS. Official checkout не изменялся.

## Количественная оценка

После завершения генерации AllTracker независимо трекает RGB-результаты и реальное продолжение с одинаковыми 38 исходными точками. Для generated добавлен observed anchor; для real используется observed `t0` + десять кадров 64…73. Координаты reported в 640×480; исходное преобразование 720×480 развёрнуто. Времена оценки 0,2…2,0 с. Generated tracks линейно интерполированы с 8 Гц на 5 Гц; visibility требуется на обоих концах интервала. При отсутствии доступного конечного кадра FDE = «—», а не 0.

Все сравнения на общей маске используют **одни и те же material points**. Первая таблица использует пересечение A/B/C; отдельная таблица ниже сохраняет собственное A/B пересечение. Coverage опубликована вместе с ошибкой. Confidence трекера не доказывает семантическую идентичность: на распавшемся стакане можно отследить фон, поэтому numeric ADE интерпретируется совместно с ручным просмотром всех 49 кадров.

| Вариант | Robot → запрошенный B control: ADE/FDE px, пары | Cup → control: ADE/FDE px, пары | Cup → real ADE px, доступные пары | MAE в общей motion ROI | Local warp MAE, 5 Гц |
|---|---|---|---|---|---|
{chr(10).join(lines)}

В первом столбце движения A оценивается относительно **запрошенной движущейся траектории B**, чтобы неподвижный робот не получал хороший результат только за совпадение с собственным неподвижным control. Ошибка относительно собственного control отдельно сохранена в JSON. Cup→real здесь имеет индивидуальные visibility masks, поэтому её строки нельзя ранжировать как сравнение на одинаковых точках; соответствующие common-mask значения также сохранены.

| Чистое A/B сравнение | Robot → B: ADE/FDE px, общие пары | Ошибка относительного cup–gripper движения, px | Валидные времена |
|---|---|---|---|
{chr(10).join(pair)}

Относительное движение измерено как разность вектора «средняя cup точка минус средняя gripper точка» и такого же вектора в control. В обоих вариантах берётся один и тот же surviving subset: минимум три cup точки и две gripper точки. Это 2D correspondence proxy, не доказательство физического 3D контакта, отсутствия скольжения или приложенной силы. C использует отдельное тройное пересечение и не включён в чистую A/B contact таблицу.

| Вариант | PSNR всей сцены, dB | MAE всей сцены | Laplacian variance | Valid flow fraction в ROI |
|---|---|---|---|---|
{chr(10).join(fullquality)}

Реальное продолжение: Laplacian variance {number(metric['real_image_quality']['laplacian_variance'])}; local warp MAE {number(metric['real_image_quality']['temporal_5Hz_local_warp']['mean_warp_MAE_0_255'])}. Общая motion ROI занимает {metric['variants']['robot_coupled']['image_quality']['shared_requested_motion_ROI_fraction']*100:.2f}% изображения. Для warp используется Farneback с forward/backward residual <1 px и явной долей валидного flow. RGB результатов линейно сэмплирован на общую сетку 5 Гц; такая интерполяция может сглаживать артефакты. Полный PSNR/MAE зависят от статического фона и несовпадения прогноза с реальностью; низкий warp error также поощряет неподвижность. Это descriptive single-clip метрики, без FVD и без вывода о perceptual или distribution-level качестве.

Ранее уже отдельно измерены MolmoMotion→real ADE 193,57 / FDE 224,48 px и ошибка его жёсткого приближения: средний residual 73,5 мм. Эти ошибки не исправляются автоматическим добавлением роботной кинематики. Прежний native no-control ролик и метрики доступны в [предыдущем исследовании](../das_wanfun/README.md); его reference/prior отличаются от A/B/C, поэтому здесь он не объявлен matched baseline.

## Визуальная оценка и вывод

{chr(10).join(visual)}

{review['interpretation_ru']}

Все 49 кадров каждого ролика доступны для проверки: [A](robot_static/all_49_generated_frames.png), [B](robot_coupled/all_49_generated_frames.png), [C](robot_coupled_initial_only/all_49_generated_frames.png). Рубрика включает движение звеньев, узнаваемую форму и рисунок синего стакана, видимую связь с захватом, прозрачность/распад, остаточные силуэты и стабильность фона. Результат не оценен как успешное реалистичное продолжение лишь на основании того, что control кинематически связан.

## Ресурсы и воспроизведение

Локальная RTX 4070 12 GB; BF16; official sequential CPU offload; FlowMatch Euler, guidance 6; TeaCache 0,10; 720×480, 49 кадров, 8 FPS. Каждый GPU-запуск последовательный, без остановки посторонних экспериментов. WSL memory limit отличается от физических 32 GB Windows. Таблица содержит process RSS, а не обещанный общий peak всех программ.

| Вариант | Generation active time, с | CUDA allocated peak, GiB | CUDA reserved peak, GiB | Process RSS peak, GiB |
|---|---|---|---|---|
{chr(10).join(resources)}

Полные commands, config, environment, logs, stage/resource receipts и SHA256 исходов сохранены рядом с каждым MP4. Выбор GPU и checkpoint не менялся между вариантами. Для hidden background применён встроенный imagegen, итоговый asset и точный prompt сохранены в `observed/`; fallback CLI для image editing не использовался.

Рабочая структура: checkout ветки в `/mnt/f/AIRI_task/das_robot_cup_20261003`, веса SAM2.1 и Wan Control в `/mnt/f/AIRI_task/models`, официальный DaS в `/mnt/f/AIRI_task/third_party/DiffusionAsShader-Wanfun`, оригинальный AllTracker в `/mnt/f/AIRI_task/molmo-motion/data_generation/third_party/alltracker`. Native source shard лежит в `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`. Получение весов и окружение описаны в прежнем DaS отчёте; checkpoint receipts фиксируют revision. Скрипты используют эту явно указанную локальную структуру. Операции генерации отказываются перезаписывать готовые ролики; для нового повторения создайте новый checkout/каталог результатов.

```bash
BASE=/mnt/f/AIRI_task
RUN=$BASE/das_robot_cup_20261003
$BASE/.venv/bin/python $RUN/scripts/das_robot_observed.py
$BASE/.venv/bin/python $RUN/scripts/das_robot_geometry.py
$BASE/.venv/bin/python $RUN/scripts/das_robot_masks.py
# imagegen_empty_robot_plate.png уже включён с точным prompt/receipt.
$BASE/.venv/bin/python $RUN/scripts/das_robot_prepare.py
# Сначала визуально просмотреть paired_control_review.png и маски.
# Перед run_pair записать фактический visual_review в control_validation.json;
# generation_ready=true допускается только после нового просмотра.
$BASE/.venv/bin/python $RUN/scripts/das_robot_run_pair.py
$BASE/.venv/bin/python $RUN/scripts/das_robot_run_aux.py
$BASE/.venv/bin/python $RUN/scripts/das_robot_evaluate.py
# qualitative_review.json — ручной просмотр реальных результатов, не автотест.
$BASE/.venv/bin/python $RUN/scripts/das_robot_report.py
$BASE/.venv/bin/python $RUN/scripts/das_robot_verify.py
$BASE/.venv/bin/python $RUN/scripts/das_artifact_manifest.py --root $RUN/runs/berkeley_ur5_molmomotion/cup/das_robot_cup --verify
```

`verification.json` проверяет frozen inputs, временную сетку, ограничения суставов, proper rotations, сохранение формы звеньев, связь стакана с flange, формат всех MP4, успех реальной генерации, параметры пары и наличие оценки/ручного просмотра. Его PASS означает корректное выполнение и сохранение эксперимента; он **не означает успешную гипотезу или реалистичный результат DaS**.
'''
    (OUT/'README.md').write_text(text,encoding='utf8')
    print('Report written',OUT/'README.md')

if __name__=='__main__':main()
