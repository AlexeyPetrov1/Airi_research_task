"""Record the assistant's actual built-in image inspection from this session.

This records manual observations; it does not run an automated vision evaluator.
Do not reuse these observations as evidence for unseen images from another run.
"""
from datetime import datetime,timezone
from fmb_wrist_prepare import write,sha
from fmb_wrist_v4_diagnose import OUT

observations={
    'wrist_2/viz/provisional_robot_comparison_20.png':
        'At2s seven of eight shown original H3 and robot forecasts are off-image above the red cap. Robot arrows are shorter and the remaining displayed point is closer; large endpoint mismatch remains.',
    'wrist_1/viz/provisional_robot_comparison_20.png':
        'Original H3 has vertically stretched, largely off-image trajectories. All eight shown robot forecasts remain inside the image but shift right onto gripper/board instead of the red target.',
    'wrist_2/viz/provisional_robot_errors.png':
        'Robot3D_est error stays below original model and Static after the early portion. Its2D advantage over CV in the middle reverses late; endpoint image error is about70px versus about35px for CV.',
    'wrist_2/viz/comparison_20.png':
        'Final six-panel comparison retains the same robot endpoint limitation. Robot and bounded hybrid each have7/8 displayed off-image points; cleaned H3 and H1 have8/8. No visually accurate2s endpoint is demonstrated.',
    'wrist_1/viz/comparison_20.png':
        'Robot and hybrid stay in image but drift right away from target; hybrid is closer. Native H1 is less numerically catastrophic than original H3 but7/8 displayed points are off-image at2s. Cleaned H3 has8/8 off-image.',
    'wrist_2/viz/errors.png':
        'New H3 remains worse than original in both dimensions; H1 almost follows original/Static. Robot/hybrid3D_est curves are lower and nearly overlap. CV retains better final2D error than robot despite worse overall2D ADE.',
    'wrist_1/viz/errors.png':
        'H1 removes the enormous original/cleaned H3 projection spikes, while staying close to Static in3D_est. Robot/hybrid errors are substantially lower, but still rise at the end. Full axes retain all large errors.',
    'wrist_2/viz/comparison_10.png':
        'At1s robot and hybrid displayed forecasts are all in image, mostly near the cap with visible offsets. Original H3, cleaned H3 and H1 each show7/8 off-image, unlike the actual reference points.',
    'wrist_1/viz/comparison_10.png':
        'At1s robot/hybrid stay inside the frame and nearer the red face, though several forecasts lie on gripper/board. Original H3 has5/8, cleaned H3 has6/8 and H1 has3/8 displayed off-image points.',
    'wrist_2/viz/full_neural_outputs.png':
        'All24 cleaned H3 trajectories are temporally flat over all30 steps. H1 has small early changes and becomes flat; all32 steps are visible. This does not demonstrate learned nontrivial robot motion.',
    'wrist_1/viz/full_neural_outputs.png':
        'Cleaned H3 groups behave inconsistently: some nearly flat, another drifts strongly inY/Z with Z crossing zero. H1 is mostly stationary after minor early changes. Full30/32-step extents preserve these failures.'
}

if __name__=='__main__':
    import json
    assert json.loads((OUT/'inference_status.json').read_text())['status']=='COMPLETE'
    write(OUT/'builtin_visual_review.json',{
        'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'method':'Actual built-in assistant view_image inspection; manual session observations recorded with artifact hashes',
        'external_HF_VLM_used':False,'future_blinded':False,
        'scope':'One already studied episode; comparisons show8/24 points, numerical errors and native plots cover all24. Green low-texture correspondences are estimates, not certified material point identities.',
        'reviewed_images':[{'image':p,'sha256':sha(OUT/p),'observation':note} for p,note in observations.items()],
        'overall_assessment':'Встроенным просмотром проверены промежуточные и итоговые изображения. Прогноз по прошлым TCP-позам и bounded hybrid заметно ближе в средней части горизонта, но через2s остаётся большой drift: wrist_2 выходит за верхнюю границу, wrist_1 смещается вправо на захват/доску. H1 на wrist_1 устраняет крупные нейронные выбросы и приближается к Static, оставаясь в основном статичным; на wrist_2 выигрыша нет. H3 cleanup не даёт устойчивого улучшения обоих ракурсов. Численный выигрыш не подтверждает точность физической геометрии, материальные соответствия всех24 точек или обобщение на новые эпизоды.'
    })
    print('Recorded actual built-in inspection of11 images',flush=True)
