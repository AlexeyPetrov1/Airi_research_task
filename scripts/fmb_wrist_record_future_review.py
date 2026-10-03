"""Serialize the assistant's completed built-in image review, preserving first judgments."""
import json
from datetime import datetime, timezone
from fmb_wrist_prepare import RUN, write, sha

assert not (RUN/'builtin_visual_review_future.json').exists(), 'Keep final review immutable'
blind_path=RUN/'builtin_blinded_review.json'
blind=json.loads(blind_path.read_text(encoding='utf-8'))
images=[dict(row, review_phase='before_method_map_and_metrics') for row in blind['reviewed_images']]

def seen(path, assessment):
    images.append({'image':path, 'sha256':sha(RUN/path),
                   'review_phase':'after_method_map_and_metrics', 'assessment':assessment})

seen('wrist_2/viz/forecast_errors.png',
     'A has a large persistent absolute3D offset while looking much better in2D. B/C overlap Static closely; C CV has smaller2D error but larger3D_est error. B CV projections grow extremely large. Ranking differs between2D and3D_est.')
seen('wrist_2/viz/full_30step_xyz.png',
     'All24 forecasts over the complete30-step range are nearly flat in each of A/B/C. A has substantially larger depth than sensor-based branches. This supports a near-static model forecast, not successful prediction of the measured downward world displacement.')
seen('wrist_2/viz/reference_hand_eye_sensitivity.png',
     'Frozen observed-bootstrap reference differences rise over the horizon, with displayed paths below roughly14mm. This descriptive envelope is not a confidence interval or physical calibration certificate.')
seen('wrist_1/viz/full_30step_xyz.png',
     'Different subsets of points diverge strongly: some Z paths rise toward0.55-0.58m, others decrease toward0.05m or change modestly. Large Y drifts and heterogeneous XYZ paths support viewpoint/group instability and fail to resemble a coherent rigid-target forecast.')
seen('wrist_1/viz/forecast_errors.png',
     'Model3D_est error grows to about343mm at2s, well above Static/CV throughout the horizon. The2D curve has a large projection spike above7000px around1.6s; these off-image projections remain in numerical metrics.')
seen('wrist_future_world_motion.png',
     'Independent reference displacement curves from the two wrists have similar dominant downward Z motion, ending near-90mm. Smaller XY disagreement remains. This supports short relative-motion consistency despite failed absolute cloud alignment.')
seen('side_future_motion_crosscheck.png',
     'Side1 visible-centroid motion magnitude roughly follows wrist2 reference and ends near83mm versus about90mm. Side2 is irregular and its small endpoint cannot be trusted as target material-point motion.')
seen('side_1/viz/side_future_color_control.png',
     'Visible red peg moves downward toward insertion in the fixed view. The colored mask follows the visible surface in sampled frames126/127/136/146, with top clipping at126 and changing visible contour. This is an approximate centroid control.')
seen('side_2/viz/side_future_color_control.png',
     'Only a small red fragment at the upper image boundary is visible in the four samples. Persistent clipping and changing visible surface make centroid motion an unreliable estimate of the complete target trajectory.')

overall=(
    'Встроенный просмотр подтвердил: на wrist_2 прогнозы A/B/C почти статичны в camera_t0. '
    'Для primary C ADE3D_est составляет42.74mm против42.84mm у Static и74.94mm у CV; '
    'выигрыш0.11mm у Static практически мал и не является убедительным улучшением. '
    'CV лучше в2D:28.49px против109.93px у C. '
    'A выглядит ближе к reference в2D, но исходное расхождение3D равно590.76mm, '
    'а displacement-only ADE46.54mm хуже Static42.76mm. '
    'На wrist_1 C даёт несогласованные между группами траектории и ADE3D_est193.20mm '
    'против40.95mm у Static и56.72mm у CV. '
    'В reference не обнаружено явного выхода выбранных ID на фон в трёх просмотренных кадрах; '
    'низкая текстура и перекрытия не позволяют подтвердить точную материальную идентичность. '
    'Две wrist-реконструкции относительного перемещения согласуются гораздо лучше абсолютных облаков '
    '(median3.34mm, endpoint4.76mm), side_1 приблизительно поддерживает движение; '
    'side_2 сильно обрезает объект и не даёт надёжной motion-проверки. '
    'Эти выводы относятся к estimated reference при неопределённых K/X/depth scale, а не к физическому GT.'
)
write(RUN/'builtin_visual_review_future.json', {
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'reviewer':'Root assistant via built-in view_image',
    'external_HF_VLM_used':False,
    'HF_flag_scope':'Quality assessment only; native MolmoPoint grounding and MolmoMotion remain part of the experiment',
    'first_blinded_review':{'path':blind_path.name, 'sha256':sha(blind_path),
                            'recorded_utc':blind['recorded_utc'],
                            'method_map_read_before_record':blind['method_map_read_before_record'],
                            'quantitative_metrics_read_before_record':blind['quantitative_metrics_read_before_record']},
    'unblinding_maps':{camera:json.loads((RUN/camera/'evaluation/blinding_map.json').read_text(encoding='utf-8'))
                      for camera in ['wrist_2','wrist_1']},
    'reviewed_images':images, 'point_reviews':blind['point_reviews'],
    'overall_assessment':overall,
    'limits':blind['limits'],
    'serialization_only':'These judgments record images actually viewed by the assistant in the original session; running this serializer does not perform a new image assessment.'
})
print('Completed built-in future review recorded; first blinded receipt preserved', flush=True)
