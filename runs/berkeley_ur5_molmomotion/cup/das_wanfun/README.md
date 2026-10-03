# Berkeley cup: MolmoMotion → DaS Wanfun

[Полный исследовательский отчёт](../../../../report/das_wanfun_cup.md).

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
