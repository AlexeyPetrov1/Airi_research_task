"""Save the completed visual decision, combined viewer and selected full arrays."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from berkeley_arc_expansion import ROOT, DEST, scene_data, annotated, sha
from berkeley_evaluate import save_rgb, project, ALIGNMENT
from berkeley_temporal_diagnostics import write

SELECTED="late_036_lift005"


def main():
    primary=json.loads((DEST/"results.json").read_text())
    phase=json.loads((DEST/"phase_schedule/results.json").read_text())
    data=dict(variants=list(primary['cup']['methods'])+list(phase['cup']),default_variant=SELECTED,scenes={})
    summary={}
    for name in ("cup","bottle"):
        s,K,history,_,_=scene_data(name)
        uv={k:v for k,v in np.load(DEST/name/"projected_2d.npz").items()}
        phase_uv=dict(np.load(DEST/"phase_schedule"/name/"projected_2d.npz"))
        uv.update(phase_uv)
        data['scenes'][name]=dict(ids=s['ids'].tolist(),uv0=s['uv0'].tolist(),gt=s['gt'].tolist(),
            mask=s['mask'].tolist(),uv={k:v.tolist() for k,v in uv.items()})
        selected=np.load(DEST/"phase_schedule"/name/"full_30_steps.npz")[SELECTED]
        folder=DEST/"selected"/name;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/"postprocessed_future_3d.npy",selected)
        np.save(folder/"point_ids.npy",s['ids'])
        np.save(folder/"K.npy",K)
        np.save(folder/"projected_2d.npy",phase_uv[SELECTED])
        np.testing.assert_allclose(project(selected[:,ALIGNMENT],K),phase_uv[SELECTED],atol=1e-12,rtol=0)
        panels=[]
        for t in (0,4,9):
            panels.append(np.concatenate([annotated(s,uv[k],t,np.arange(24),k)
                for k in ["reference_0333",SELECTED,"CV_3D"]],axis=1))
        save_rgb(folder/"all24_comparison.png",np.concatenate(panels))
        summary[name]=dict(episode_id=primary[name]['episode_id'],t0=primary[name]['t0'],K=K.tolist(),
            shape=list(selected.shape), metrics=phase[name][SELECTED],
            raw_prediction_source=f"../../berkeley_ur5_improvement_v1/CASE-AUGE/{name}/predictions/future_3d.npy",
            point_ids=s['ids'].tolist(),postprocessed_prediction_sha256=sha(folder/"postprocessed_future_3d.npy"))
    template=(ROOT/"scripts/berkeley_arc_gallery.html").read_text(encoding='utf8')
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (DEST/"gallery_phase.html").write_text(template.replace('__EXPERIMENT_DATA__',payload),encoding='utf8')
    write(DEST/"selected/metadata.json",dict(variant=SELECTED,postprocessing_only=True,raw_model_changed=False,
        shared_K=True,formula="alpha(t)=1/3+(0.36-1/3)*smoothstep(clamp(t-1,0,1)); Y-=0.005*same_smoothstep",
        first_second_identical_to_one_third=True,endpoint_scale_increase_percent=8.,endpoint_camera_up_m=.005,
        gravity_direction_known=False,accuracy_improvement_on_both_scenes=False,scenes=summary))
    review=dict(completed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),
        shared_candidate=SELECTED, shared_candidate_family='phase_schedule',
        selection_purpose="Small, shared farther-and-higher change with less early amplification; not best measured accuracy",
        criteria_order=['real RGB and point correspondence','direction and lift/place timing','arc extent','P8 and across-group coherence','secondary metrics'],
        reviewed=dict(full_curve_grids='both all24_candidate_grid.png; all 11 constant/lift candidates',
            decoded_all10_movies=['cup/controls_group_00..02','bottle/controls_group_00..02',
                'phase_schedule/cup/timing_group_00..02','phase_schedule/bottle/timing_group_00..02'],
            height_contacts=['cup/height_group_00_contact.png','bottle/height_group_00_contact.png'],
            scope='All 24 IDs and all ten real future times for primary constant/late choices and CV; not every generated candidate movie watched in full'),
        cup=dict(groups={
            '0':'Late placing bend to the right is already premature. Expansion raises/extends it but cannot restore the observed vertical lift.',
            '1':'Vertical lift remains coherent. Constant scale leads the cup upward early; delayed expansion leaves the first second intact.',
            '2':'Same early vertical lead. Larger late extent can put endpoints above the reference; expansion is not uniformly beneficial.'},
            conclusion='Use the delayed small expansion to demonstrate the requested shape change; retain original x1/3 and CV as accuracy controls.'),
        bottle=dict(groups={
            '0':'Rightward arc begins too soon. Extra height improves some final positions; delayed expansion avoids increasing the first-second lead.',
            '1':'Similar early bend toward pot; endpoints can move closer with the small height shift.',
            '2':'Wider curve and larger endpoint discrepancy persist; raising scale hurts this group despite gains in the other groups.'},
            conclusion='Shared late expansion modestly improves global endpoint error, while average error remains slightly worse than x1/3.'),
        rejected=dict(scale_040_and_042='Amplify premature motion and widen arcs too strongly for a slight shared change',
            constant_scale_036_lift005='Same final extent as selected schedule but visibly larger early lead',
            independent_per_scene_tuning='Not used; both scenes share the same K and rule'),
        limitations=['AugE K is a calibration hypothesis','Camera-up is not world height','Tracker reference is not manual ground truth',
            'Greedy genuine model forecasts are reused; this is postprocessing','H3 5 FPS versus documented F30 15 FPS remains a temporal shift',
            'Same two previously inspected scenes, not independent test evaluation','Common smoothing Z is retained from prior CASE-AUGE ray-only control'])
    write(DEST/"visual_review.json",review)
    lines=['# Небольшое расширение траекторий Berkeley', '',
        'Общий осторожный вариант: **после +1 с плавно увеличить α с 1/3 до 0,36 и добавить до 5 мм по −Y камеры**. До +1 с прогноз совпадает с прежним ×1/3. К +2 с он дальше примерно на 10–11 пикселей и выше на 10 пикселей в обеих сценах. Это выбранный вариант для небольшого изменения дуги; улучшения точности на обеих сценах нет.', '',
        '[Открыть интерактивное сравнение всех 15 вариантов и трёх бейзлайнов](gallery_phase.html). Открывается локально в браузере после клонирования репозитория; GitHub показывает HTML как исходный код. Один переключатель меняет сразу обе сцены. Доступны все 24 точки, три отдельные P8 группы, десять настоящих моментов и воспроизведение 5 FPS.', '',
        '[Полный понятный разбор процесса и выводов](research_story.md) · [Визуальная оценка](visual_review.json) · [Выбранные массивы и параметры](selected/metadata.json).', '',
        '| Сцена | Метод | ADE, px ↓ | FDE, px ↓ |', '|---|---|---:|---:|']
    for name in ('cup','bottle'):
        methods={k:primary[name]['methods'][k] for k in ['reference_0333','scale_036_lift005','lift_010','CV_3D','object_translation_3D']}
        methods[SELECTED]=phase[name][SELECTED]
        for key,m in methods.items():lines.append(f"| {name} | {key} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} |")
    lines+=['', 'K в обеих сценах побитово одинаковый: fx = fy = 434,5584411621094, cx = 320, cy = 240. Точное значение из FOV до сохранения float32 — 434,5584412271572. Используются сохранённые настоящие CASE-AUGE predictions всех трёх групп. Новый inference и обучение не выполнялись; SAM, RGB, history, depth и point IDs не менялись.', '',
        'Статика, per-point CV и общее медианное перемещение построены только по наблюдавшимся данным. CV остаётся значительно ближе к reference на этих коротких окнах. Лучшее FDE отдельных вариантов не означает лучшую траекторию по всем моментам: одинаковые концы constant/late вариантов дают одинаковый FDE, но разную ошибку в середине.', '',
        'Сначала заморожены 11 вариантов масштаба/высоты, затем после визуальной диагностики преждевременного движения — четыре одинаковых для обеих сцен временных правила. Это исследовательская проверка уже просмотренных сцен; численное подбирание коэффициента по будущему не применялось. Выбор и гипотеза используют будущие evaluation кадры и не являются независимым test split.', '',
        '| Сцена | Данные | Все 24 точки | Три группы и реальные кадры |', '|---|---|---|---|',
        '| cup | episode 10, t0=63 | [сравнение](selected/cup/all24_comparison.png), [3D NPY](selected/cup/postprocessed_future_3d.npy) | [группа 0 MP4](phase_schedule/cup/timing_group_00.mp4), [группа 1 MP4](phase_schedule/cup/timing_group_01.mp4), [группа 2 MP4](phase_schedule/cup/timing_group_02.mp4) |',
        '| bottle | episode 9, t0=47 | [сравнение](selected/bottle/all24_comparison.png), [3D NPY](selected/bottle/postprocessed_future_3d.npy) | [группа 0 MP4](phase_schedule/bottle/timing_group_00.mp4), [группа 1 MP4](phase_schedule/bottle/timing_group_01.mp4), [группа 2 MP4](phase_schedule/bottle/timing_group_02.mp4) |', '',
        'Все 30 будущих steps сохранены; оценка берёт indices [2,5,…,29] и реальные моменты +0,2…+2,0 с. Всего 30 MP4: 18 scale/height/control и 12 phase. В каждой папке есть полные XYZ/UV NPZ, contact sheets и монтажи всех десяти декодированных кадров. Для монтажей время идёт четырьмя столбцами; на каждый блок времени три вертикальные строки соответствуют трём вариантам из render_receipts.json. Сами MP4 сохраняют полный исходный кадр.', '',
        '[Все метрики 11 основных вариантов](results.json), [четырёх временных правил](phase_schedule/results.json), [замороженный основной протокол](protocol.json), [протокол продолжения](phase_schedule/protocol.json), [проверки](verification.json), [ресурсы GitHub/Hugging Face](resource_provenance.json).', '',
        'Метрики вторичны: native 2D ADE/FDE, PWT<5/10/20/40px, E(t), направление и скорость, длина пути, конечное перемещение, 3D pair-distance drift. 3D_est использует tracker и native Z в гипотезе K и не является calibrated ground truth. Предсказания вне кадра не отбрасываются из оценки.', '',
        'Для проверки сохранённых результатов:', '', '```bash', 'python -m pytest tests/test_berkeley_arc_expansion.py -q',
        'python scripts/berkeley_arc_expansion.py --stage verify', 'python scripts/berkeley_arc_verify.py', '```', '',
        'Для повторения в новой папке (старые runs не перезаписываются):', '', '```bash',
        'python scripts/berkeley_arc_expansion.py --stage generate --output-dir runs/berkeley_arc_repeat',
        'python scripts/berkeley_arc_phase.py --parent-run runs/berkeley_arc_repeat', '```', '',
        '[Предыдущее исследование и источник настоящих модельных прогнозов](../berkeley_ur5_improvement_v1/research_story.md).']
    (DEST/"README.md").write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'selected':SELECTED,'gallery':'gallery_phase.html','scenes':list(summary)}),flush=True)


if __name__=='__main__':main()
