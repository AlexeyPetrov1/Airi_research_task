# Дополнительные численные результаты

Главные выводы сформулированы по изображениям в [полном рассказе](research_story.md) и [визуальном отчёте](visual_review_final.json). Ниже все 24 неизменных ID; строки H1 и oracle отделены по размеру выборки и использованию будущего. Значения 3D_est зависят от оценённой K и приведены только в JSON.

| Сцена | Вариант | ADE 2D, px | FDE 2D, px |
|---|---|---:|---:|
| cup | BASELINE-U/original_physical | 193.571 | 224.485 |
| cup | BASELINE-U/displacement_one_third | 46.917 | 69.211 |
| cup | BASELINE-U/step_equals_frame | 54.436 | 116.072 |
| cup | CASE-AUGE_actual | 152.304 | 174.449 |
| cup | CASE-AUGE_one_third | 28.111 | 36.227 |
| cup | CASE-NOK_actual | 131.621 | 132.321 |
| cup | CASE-NOK_one_third | 19.004 | 18.670 |
| cup | CASE-NOK-STRICT_actual | 125.927 | 118.177 |
| cup | CASE-NOK-STRICT_one_third | 18.699 | 22.277 |
| bottle | BASELINE-U/original_physical | 223.409 | 194.015 |
| bottle | BASELINE-U/displacement_one_third | 47.950 | 36.494 |
| bottle | BASELINE-U/step_equals_frame | 103.084 | 181.552 |
| bottle | CASE-AUGE_actual | 218.392 | 201.396 |
| bottle | CASE-AUGE_one_third | 44.650 | 34.255 |
| bottle | CASE-NOK_actual | 208.823 | 222.507 |
| bottle | CASE-NOK_one_third | 42.275 | 44.254 |
| bottle | CASE-NOK-STRICT_actual | 214.116 | 228.994 |
| bottle | CASE-NOK-STRICT_one_third | 45.617 | 50.344 |

Контроли вычислены в геометрии соответствующей ветки, из её H3, на том же 2D reference:

| Сцена | Геометрия | Контроль | ADE 2D, px | FDE 2D, px |
|---|---|---|---:|---:|
| cup | CASE-AUGE | static | 46.114 | 83.608 |
| cup | CASE-AUGE | constant_velocity | 2.906 | 7.374 |
| cup | CASE-NOK | static | 46.114 | 83.608 |
| cup | CASE-NOK | constant_velocity | 2.906 | 7.374 |
| cup | CASE-NOK-STRICT | static | 46.114 | 83.608 |
| cup | CASE-NOK-STRICT | constant_velocity | 2.735 | 6.476 |
| bottle | CASE-AUGE | static | 60.809 | 102.199 |
| bottle | CASE-AUGE | constant_velocity | 11.045 | 11.934 |
| bottle | CASE-NOK | static | 60.809 | 102.199 |
| bottle | CASE-NOK | constant_velocity | 11.045 | 11.934 |
| bottle | CASE-NOK-STRICT | static | 60.809 | 102.199 |
| bottle | CASE-NOK-STRICT | constant_velocity | 10.935 | 10.383 |

Oracle: коэффициент подбирается по будущей 3D_est траектории. Это диагностика, не результат независимого предсказания.

| Сцена | Вариант | ADE 2D, px | FDE 2D, px |
|---|---|---:|---:|
| cup | fixed_one_third | 46.917 | 69.211 |
| cup | ORACLE_same_scene | 24.145 | 63.240 |
| cup | transfer_from_bottle | 25.884 | 61.623 |
| bottle | fixed_one_third | 47.950 | 36.494 |
| bottle | ORACLE_same_scene | 27.788 | 51.479 |
| bottle | transfer_from_cup | 27.188 | 58.036 |

H1: только первые восемь точек чашки, полный raw ответ содержит 32 шага; оценка на тех же десяти физических моментах.

| Вариант | ADE 2D, px | FDE 2D, px |
|---|---:|---:|
| BASELINE-U | 198.012 | 227.299 |
| CASE-H1_actual | 227.850 | 280.512 |
| CASE-H1_one_third | 97.467 | 147.860 |

[Полные числа и источники каждой строки](summary.json).
