"""Build the final source-linked report from completed measured experiments."""
import argparse
import json
from das_full_motion_diagnose import OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--chosen',required=True);a=p.parse_args()
    names=['H1_group00_2s_no_prior','H2_group00_6s_no_prior','H3_group00_6s_prior025',
        'H4_no_trajectory_control','H5_group00_6s_whole_robot_prior025']
    if a.chosen not in names:names.append(a.chosen)
    review=json.loads((OUT/'visual_review.json').read_text())
    assert review['chosen_variant']==a.chosen
    matched=json.loads((OUT/'matched_comparisons.json').read_text())
    robot=json.loads((OUT/f'{a.chosen}_robot_comparison.json').read_text())
    records={name:json.loads((OUT/name/'motion_metrics.json').read_text()) for name in names}
    resources={name:json.loads((OUT/name/'resource_usage.json').read_text()) for name in names}
    def fmt(v):return '—' if v is None else f'{v:.2f}'
    rows=[]
    for name in names:
        m=records[name]['generated_to_raw_forecast'];c=json.loads((OUT/name/'config.json').read_text())
        timing='2 с + hold' if c['timing']=='physical_2s' else '6 с'
        rows.append(f"| [{name}]({name}/generated_seed42.mp4) | {timing} | {c['photographic_guide_strength']:.2f} | {fmt(m['ADE_px'])} | {fmt(m['FDE_px'])} | {m['visible_pairs']}/392 | {resources[name]['wall_seconds']:.1f} |")
    timing=matched['matched_timing'];prior=matched['matched_prior'];control=matched['matched_control']
    body=robot['variants'];selected=review['variants'][a.chosen]
    bodyrows=[]
    for name,d in body.items():
        m=d['shared_three_proximal_to_requested_motion'];dis=d['shared_three_proximal_displacement']
        bodyrows.append(f"| [{name}]({name}/robot_tracking_overlay.mp4) | {fmt(m['ADE_px'])} | {m['visible_pairs']} | {fmt(dis['mean_px'])} | {fmt(dis['max_px'])} |")
    outcomes='\n\n'.join(f"**{name}.** {review['variants'][name].get('acceptance_ru',review['variants'][name]['acceptance'])}" for name in names)
    config=json.loads((OUT/a.chosen/'config.json').read_text())
    chosen_prior=json.loads((OUT/a.chosen/'prior_latent_audit.json').read_text())
    chosen_resource=resources[a.chosen]
    real=records[names[0]].get('generated_to_real_physical_2s')
    realtext=(f"После завершения всех генераций H1 отдельно оценён на реальном продолжении 0,2–2,0 с: ADE {fmt(real['ADE_px'])} px, FDE {fmt(real['FDE_px'])} px, {real['visible_pairs']} пригодных пар. Из-за потери трека эта оценка ограничена оставшимися видимыми соответствиями. [H1 metrics](H1_group00_2s_no_prior/motion_metrics.json), [generated vs real](physical_generated_vs_real_2s.mp4)." if real else 'Оценка растянутого шестисекундного видео проводится относительно фазы прогноза, а не реального шестисекундного продолжения.')
    text=f'''# Полная дуга MolmoMotion и движение руки UR5 → DaS

Исследование начато 3 октября 2026; итог оформлен 4 октября по Москве. Ветка `codex/das-full-motion-20261003`.

**Выбранный результат: [видео {a.chosen}]({a.chosen}/generated_seed42.mp4).** {selected.get('acceptance_ru',selected['acceptance'])}

[H2 / H3 / новое движение руки рядом]({a.chosen}_body_comparison.mp4) · [старый F и новый результат](before_after_full_arc.mp4) · [все проверки](verification.json) · [файлы и SHA256](artifact_manifest.json).

![Все 49 кадров выбранного результата]({a.chosen}/all_49_generated_frames.png)

## Причина прежнего подъёма на месте

В `das_reference_repair` пространственная траектория MolmoMotion была заменена интерполяцией между реальными кадрами 63 и 73. MolmoMotion задавал только скалярную фазу; endpoint, RGB-prior и prompt описывали подъём. Старый F сохранён без изменений как вариант с известным будущим endpoint. Сравнение F с новым видео наглядное, но не парное: различаются геометрия, timing и тип prior.

В левой колонке исходного `fixed_comparison.mp4` показаны первые восемь baseline-точек, **group_00**. Три независимые P8-группы расходятся. Усреднение 24 точек одним Kabsch даёт средний RMS 73,50 мм; fit group_00 — 1,61 мм. Его средняя ошибка проекции к исходному raw-прогнозу — 0,75 px. Для воспроизведения выбранной пользователем дуги используется group_00. [Диагностика](forecast_diagnostics.json), [три группы и median](group_paths.png), [обычный и robust Kabsch](rigid_methods_comparison.mp4). All24 и robust24 здесь CPU-диагностика, без утверждения об их диффузионной генерации.

Новые редкие реальные кадры на вход MolmoMotion не добавлялись. Наблюдаемые кадры 61, 62, 63 имеют timestamps 12,2; 12,4; 12,6 с — уже 5 Hz, при соглашении модели 15 Hz. Изменение этой истории изменило бы сам прогноз. Здесь прогноз заморожен.

## Время и контролируемые сравнения

Исходные 30 будущих шагов плюс t0 описывают горизонт 2 с. Для вращения применяется SLERP, для переноса — линейная интерполяция. T2 проходит путь за кадры 0–16, затем держит endpoint; T4 — за 0–32; T6 — за 0–48. T4 подготовлен только на CPU. T6 — **time-stretched visualization of the predicted spatial trajectory**, с трёхкратным замедлением, а не исправленный физический прогноз. Контейнер из 49 кадров при 8 FPS имеет длину 6,125 с; между первым и последним кадром проходит 6 с. [29 проверок одинаковых поз при одинаковой фазе](cpu_verification.json).

| Вариант / фактическое видео | Движение | Prior | ADE raw, px | FDE raw, px | Пригодные пары | Время генерации, с |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Это индивидуальные оценки всех 49 кадров, с разным покрытием и, у H1, другим timing. Пропуски не заменяются нулевыми ошибками. Все метрики и видимость по времени: [comparison_metrics.json](comparison_metrics.json).

**Timing H1/H2:** сравниваются 17 одинаковых фаз исходного пути; hold исключён. На {timing['common_pairs']} общих парах ADE H1 = {fmt(timing['variants'][names[0]]['ADE_px'])}, H2 = {fmt(timing['variants'][names[1]]['ADE_px'])} px. Индивидуально пригодны {timing['individual_visible_coverage_at_same_phases'][names[0]]['usable_pairs']}/136 и {timing['individual_visible_coverage_at_same_phases'][names[1]]['usable_pairs']}/136 пар. Общая маска не покрывает конец пути из-за потери H1, поэтому paired FDE отсутствует. Визуально быстрый вариант меняет материал и форму стакана; шестисекундный перенос устойчивее. [Сравнение одинаковых фаз](timing_comparison_equal_phase.mp4).

**Prior H2/H3:** одинаковые control, seed, prompt, sampler и время. На {prior['common_pairs']} общих парах ADE {fmt(prior['variants'][names[1]]['ADE_px'])} → {fmt(prior['variants'][names[2]]['ADE_px'])} px. При этом основная рука в обоих остаётся неподвижной.

**Control H2/H4:** в обоих prior выключен; H4 получает `control_video=None`, а официальный pipeline формирует нулевые control latents. На {control['common_pairs']} общих парах ADE H2 = {fmt(control['variants'][names[1]]['ADE_px'])}, H4 = {fmt(control['variants'][names[3]]['ADE_px'])} px. [Все paired metrics](matched_comparisons.json).

{outcomes}

## Новый опыт со всей рукой

После H1–H4 выполнен отдельный запрос пользователя: управлять плечом и предплечьем, сохранив дугу H2. Файл `motion.npz` стакана и локального wrist/gripper в новом опыте **байт-в-байт совпадает с H2**. Стакан не переносится на достигнутую IK-позу: сохраняется исходное преобразование SE(3).

Используются наблюдаемые состояния суставов, маски звеньев и приблизительная регистрация камеры из t0. Номинальная цепь строится по [официальным DH-параметрам UR5](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/). Последовательный bounded IK следует желаемому flange pose с ограничением 2 rad/s. Плечо отображается между неподвижной базовой осью и движущимся локтем; предплечье — между локтем и точкой соединения с неизменным движением запястья. Используются цельные t0 RGB-силуэты и плотные image-space карты материала, чтобы не возвращать дырявую поверхность отражающего металла.

Все старые положения видимых подвижных звеньев очищены t0-only synthetic clean plate. Закреплённое основание остаётся неподвижным. Новые стороны поверхности дорисовывает диффузия. [Подготовленный guide](group00_stretched_6s_whole_robot_v1/photographic_guide.mp4) — входной синтетический сигнал, а не результат генерации. [Кинематика](whole_robot_ik_diagnostic/receipt.json), [границы регистрации](whole_robot_ik_diagnostic/registration_scope.json), [подготовка и ограничения](group00_stretched_6s_whole_robot_v1/preparation.json).

![Суставы и изменение конфигурации](whole_robot_ik_diagnostic/articulated_joint_paths.png)

Малый численный residual IK относится к принятой номинальной цепи. Ошибки исходной регистрации четырёх t0 суставов составляют 3,05; 17,07; 13,32; 10,41 px; расхождения depth около 0,14–0,18 м. Поэтому это приблизительное визуальное управление, а не заводская 3D-калибровка или план исполнения роботом. Изображение звеньев отображается 2D similarity, а запястье сохраняет единый rigid transform H2.

Независимо отслеживаются 6 материальных точек плеча/предплечья и 5 точек локального wrist/gripper. Ниже один и тот же набор общих видимых пар H2/H3/выбранного опыта. Цель для всех — новое движение руки; H2/H3 служат сохранёнными отрицательными примерами неподвижной руки.

| Видео / overlay | ADE плеча и предплечья, px | Общие пары | Среднее смещение от t0, px | Максимальное смещение, px |
|---|---:|---:|---:|---:|
{chr(10).join(bodyrows)}

Числа дополнены просмотром всех кадров. Видимость учитывает реальный выход точки из кадра; скрытые звенья не оцениваются как неподвижные. [Подробности, покрытие по времени и относительное движение cup/gripper]({a.chosen}_robot_comparison.json). Это 2D измерения видимых точек, без сертификации 3D-контакта или идентичности предмета.

## Диффузия, prior и ресурсы

Официальный [DaS Wanfun](https://github.com/IGL-HKUST/DiffusionAsShader/tree/{config['das_commit']}), commit `{config['das_commit']}`; [Wan2.1-Fun 1.3B Control на HF](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control/tree/{config['checkpoint_revision']}), revision `{config['checkpoint_revision']}`. RTX 4070 12 GB, host 32 GiB RAM, WSL limit 24 GiB, BF16, official sequential CPU offload, seed 42, Euler/shift 3, guidance 5, 30 шагов, TeaCache выключен, 720×480, 49 кадров, 8 FPS, initial-only reference (`ref_image=None`). Все текущие успешные запуски локальные; платные HF Jobs не запускались.

У H3 и опыта со всей рукой guide построен только из t0. После каждого Euler step применяется частичный spatial latent prior с корректным уровнем следующего sigma и фиксированным шумом. Первый latent якорится с силой 1.0; остальные веса и их маски сохранены. Для выбранного опыта global strength = {config['photographic_guide_strength']:.2f}. [Формула и sigma]({a.chosen}/photographic_prior_receipt.json), [config]({a.chosen}/config.json).

Выходные RGB получены **декодированием сохранённых диффузионных латентов**: после генерации кадры не подменяются фотографиями или композитами. Audit близости к guide исключает initial latent: relative RMS выбранного результата — {chosen_prior['future_all_latent_relative_RMS']*100:.2f}% по будущим латентам целиком и {chosen_prior['future_moving_region_relative_RMS']*100:.2f}% в moving region. Это мера близости к условию, не реализма или качества движения. Она не сравнима напрямую с 2,56% старого F, у которого использован будущий endpoint и другая область подсчёта. [Latent audit]({a.chosen}/prior_latent_audit.json).

Ресурсы выбранного опыта: {chosen_resource['wall_seconds']:.1f} с; CUDA allocated peak {chosen_resource['peak_cuda_allocated_bytes']/2**30:.2f} GiB, reserved {chosen_resource['peak_cuda_reserved_bytes']/2**30:.2f} GiB, RSS {chosen_resource['peak_process_rss_bytes']/2**30:.2f} GiB. Время предварительного ожидания другого GPU-процесса учитывается отдельно. По выбору пользователя другой прогон не останавливался. Неуспешные попытки и внешняя перезагрузка сохранены как инциденты, без выдачи за результаты: [VRAM](shared_gpu_incident.json), [reboot](reboot_incident.json).

## Границы оценки и воспроизведение

AllTracker Net(16), 4 итерации, 512×384, pinned checkpoint, дополнительные наблюдаемые t0 anchors. Из оценки исключён дополнительный anchor; координаты возвращаются к native 640×480. Пара требует tracker visibility и нахождения обеих точек в изображении. Трекер может переключиться на фон или деформированный объект; поэтому всем роликам сопоставлен просмотр 49 кадров. Низкий ADE сам по себе не доказывает идентичность стакана или реалистичность руки.

{realtext}

Точная дуга group_00 выходит за верх кадра и заканчивается выше/правее коричневого стакана. Это сохранено по явному выбору пользователя. Вложение одного стакана в другой не следует из этого endpoint. Причинность здесь относится к conditioning текущей генерации. Выбор эпизода/t0 унаследован из прошлого исследования, где был future screening: одна сцена и seed 42 не образуют слепой dataset benchmark. [Точная область вывода](causality_scope.json).

Старые F/H2 и исходный comparison сохранены и проверяются по SHA256. Выполнены 7 unit-проверок, 29 CPU-проверок геометрии/времени и итоговая проверка всех запрошенных вариантов, frozen inputs, точных латентов, guide cache, одинаковых initial RGB, независимых tracker receipts и ручного просмотра. [verification.json](verification.json), [назначение каждой подготовки](preparation_inventory.json), [visual_review.json](visual_review.json).

Для новой генерации по опубликованной замороженной подготовке в локальном WSL-окружении:

```bash
PY=/mnt/f/AIRI_task/.venv-das/bin/python
REPO=/mnt/f/AIRI_task/das_full_motion_20261003
$PY $REPO/scripts/das_full_motion_generate.py \\
  --name reproduced_whole_robot \\
  --preparation-subdir {Path_name(config['preparation_root'])} \\
  --strength {config['photographic_guide_strength']}
```

Генератор требует совпадения frozen SHA256 и принятого review, ждёт освобождения GPU перед загрузкой моделей, запрещает чтение реального `evaluation` и перезапись готового видео. Первому H2 audit hook добавлен позднее; отсутствие будущих RGB устанавливается по его сохранённому коду и freeze manifest. H3 повторно использовал точно тот же guide latent из сохранённой прерванной попытки, что проверяется по SHA256 и `torch.equal`. Все исходные латенты, snapshots кода, окружение, input receipts, видео, previews и измерения сохранены в этой директории.
'''
    (OUT/'README.md').write_text(text,encoding='utf8')
    (OUT/'comparison_metrics.json').write_text(json.dumps({'variants':records,'future_used_in_generation':False},indent=2)+'\n')
    print('Final report built from completed records:',a.chosen)


def Path_name(path):
    from pathlib import Path
    return Path(path).name


if __name__=='__main__':main()
