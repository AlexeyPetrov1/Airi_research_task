# Berkeley cup: MolmoMotion → DaS Wanfun

[Полный исследовательский отчёт](../../../../report/das_wanfun_cup.md).

**Исправление по замечанию пользователя:** [новое видео v6](variants/clean_reference_mask_v6/generated_molmomotion_seed42.mp4) устраняет постоянный стакан и ободок на исходном месте с 0.5 с до конца. В начале есть короткий fade; движение и узнаваемость подвижного предмета остаются несовершенными. Это расширение inference DaS с causal пустым reference и vacancy-prior, без ретуши готовых кадров. [Отчёт по шести новым вариантам](../../../../report/das_cup_dedup.md).

| Новые материалы | Ссылка |
|---|---|
| Исходный ролик / v6, все 49 кадров | [before_after_full_duration.mp4](variants/clean_reference_mask_v6/before_after_full_duration.mp4), [контактные кадры](variants/clean_reference_mask_v6/before_after_contact_sheet.png) |
| Увеличенное исходное место, все 6 с | [initial_site_before_after.mp4](variants/clean_reference_mask_v6/initial_site_before_after.mp4) |
| Real / repaired control / v6, первые 2 с | [triple_comparison.mp4](variants/clean_reference_mask_v6/triple_comparison.mp4) |
| Все 49 кадров области и полного изображения v6 | [исходное место](variants/clean_reference_mask_v6/all_49_site_repaired.png), [полный кадр](variants/clean_reference_mask_v6/all_49_full_repaired.png) |
| Метрики v6, локальная согласованность и остаточная текстура | [metrics.json](variants/clean_reference_mask_v6/metrics.json), [local_quality.json](variants/clean_reference_mask_v6/local_quality.json), [duplicate_texture_audit.json](variants/clean_reference_mask_v6/duplicate_texture_audit.json) |
| Общие и попарные маски всех вариантов | [variant_comparison.json](variant_comparison.json) |
| Параметры v6, расходы и prior receipt | [config.json](variants/clean_reference_mask_v6/config.json), [resource_usage.json](variants/clean_reference_mask_v6/resource_usage.json), [vacancy_prior_receipt.json](variants/clean_reference_mask_v6/vacancy_prior_receipt.json) |
| Визуальная и техническая проверка v6 | [qualitative_review.json](variants/clean_reference_mask_v6/qualitative_review.json), [verification.json](variants/clean_reference_mask_v6/verification.json) |
| Итоговая проверка видео, чисел отчёта и локальных ссылок | [delivery_verification.json](delivery_verification.json) |
| Causal пустой reference после исправления dark-blue/gripper | [clean_reference_640x480.png](repair_assets/clean_reference_mask_v2/clean_reference_640x480.png), [receipt](repair_assets/clean_reference_mask_v2/clean_reference_receipt.json) |
| Исходный imagegen output и точный prompt, режим edit по observed t0 | [clean_plate_imagegen.png](repair_assets/clean_plate_imagegen.png), [imagegen_prompt.txt](repair_assets/imagegen_prompt.txt) |
| Все попытки и неудачные варианты | [v1](variants/background_completion_v1), [v2](variants/background_initial_only_v2), [v3](variants/clean_background_reference_v3), [v4](variants/vacancy_prior_v4), [v5](variants/vacancy_context_v5) |

Ветка исправления: `codex/das-cup-dedup-20261003-1154`. V6 выбран по устранению постоянного дубля. На общей маске исходного ролика и v6 (93/240 пар) control ADE/FDE = 143.05/230.96 → 89.29/225.51 px, real = 29.93/86.43 → 69.11/63.32 px. Это не подтверждает физическую идентичность и общее улучшение качества. Исходная no-control абляция ниже имеет прежний reference-протокол и не является парной абляцией vacancy-prior.

Основной опыт выполнен 3 октября 2026 года: официальный DaS `Wanfun`, готовые обученные веса Alibaba PAI `Wan2.1-Fun-V1.1-1.3B-Control`, BF16, seed 42, 25 шагов, 49 кадров 720×480 при 8 FPS. Прогноз MolmoMotion переиспользован, реальное будущее не входило в управление. Движение исходного стакана выполнено плохо: он остаётся на месте, а вдоль управляющей траектории появляется отдельная копия/фрагмент.

| Материал | Ссылка |
|---|---|
| Основное сгенерированное видео, все 6 с | [generated_molmomotion_seed42.mp4](generated_molmomotion_seed42.mp4) |
| Видео без управляющей траектории, все 6 с | [generated_no_control_seed42.mp4](generated_no_control_seed42.mp4) |
| Control / с траекторией / без траектории, первые 2 с | [ablation_comparison.mp4](ablation_comparison.mp4), [контактные кадры](ablation_contact_sheet.png) |
| Обе генерации до 6 с | [ablation_full_duration.png](ablation_full_duration.png) |
| Real / dense control / generated, первые 2 с | [triple_comparison.mp4](triple_comparison.mp4) |
| Контактные кадры с реальным продолжением | [generated_contact_sheet.png](generated_contact_sheet.png) |
| Все исходные метрики и маски | [metrics.json](metrics.json), [aligned_evaluation.npz](aligned_evaluation.npz) |
| Метрики отсутствующего управления и сравнение | [metrics_no_control.json](metrics_no_control.json), [ablation_metrics.json](ablation_metrics.json) |
| Траектории и ошибки | [motion_comparison.png](motion_comparison.png) |
| Проверенный плотный control video | [control_molmomotion_720x480.mp4](control_molmomotion_720x480.mp4) |
| Контрольные кадры построения управления | [control_contact_sheet.png](control_contact_sheet.png) |
| Ошибка жёсткой аппроксимации | [rigid_fit.json](rigid_fit.json), [rigid_fit_residual.png](rigid_fit_residual.png) |
| Параметры и pinned revisions | [config.json](config.json), [checkpoint_receipt.json](checkpoint_receipt.json) |
| Реальные затраты и полный лог | [resource_usage.json](resource_usage.json), [generation.log](generation.log) |
| Затраты и лог без управления | [resource_usage_no_control.json](resource_usage_no_control.json), [generation_no_control.log](generation_no_control.log) |
| Качественная оценка и ограничения | [qualitative_review.json](qualitative_review.json) |
| Проверка конвейера | [verification.json](verification.json) |
| Размеры и SHA256 всех файлов опыта | [artifact_manifest.json](artifact_manifest.json) |

ADE/FDE generated→control = 199.02/237.89 px, generated→real = 45.95/83.43 px. Первая пара использует 187/240 видимых соответствий, вторая — 240/240; метрики общей маски находятся в JSON. Это треки исходного стакана; они не идентифицируют появляющуюся копию. PASS проверки означает воспроизводимость выполненного опыта, а не хорошее качество управления.

Контроль без траектории выполнен с нативным `control_video=None` при тех же параметрах. ADE/FDE относительно заданного движения = 186.73/210.11 px, относительно real = 43.07/75.17 px. В нём появляются человекоподобная кисть и дополнительные стаканы; исходный предмет также не переносится корректно. Управление изменяет картинку, но не улучшает движение исходного стакана в этом эпизоде. Один эпизод и один seed не дают общего вывода о модели.
