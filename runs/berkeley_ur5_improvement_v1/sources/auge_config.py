import importlib
from functools import lru_cache

from core.bridge_viewpoints import bridge_viewpoints

config = {
    # Example dataset entry mirroring the structure explained in README (see "Register the dataset in config.py").
    # - `viewpoints`: camera pose metadata consumed by replay utilities.
    # - `process_function`: callable from processing_utils that preps raw data.
    # - `out_path`: root directory for cleaned/inpainted videos (previously `inpainted_video_path`).
    # - `extend_gripper`: override for gripper reach applied during replay.
    # - `num_episodes`: per-split counts used by bulk_pipeline.py when iterating episodes.
    "example_dataset": {
        "viewpoints": [{
            "lookat": [0.0, 0.0, 0.0],
            "distance": 1.0,
            "azimuth": 0.0,
            "elevation": 0.0,
            "episodes": {
                "train": [0, 1, 2],
                "val": [0]
            }
        }],
        "process_function": "process_toto",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "use_joint_angles": True,
        "source_robot": "panda",
        "num_episodes": {
            "train": 3,
            "val": 1
        }
    },
    "toto": {
        "viewpoints": [{
            "lookat": [0.360729295, 0.106302025, 0.13910922],
            "distance": 1.03,
            "azimuth": 119.8125,
            "elevation": -18.59375,
            "episodes": {
                "train": list(range(902)),
                "test": list(range(101))
            }
        }],
        "process_function" : "process_toto",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 902,
            "test": 101
        }
    },

    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.0468699, -0.11639435, 0.60702954],
            "distance": 0.4460256912283013,
            "azimuth": 29.454542284691648,
            "elevation": -11.94610988080058,
            "camera_fov": 45,
            "episodes": {
                "train": list(range(365)),
                "val": list(range(91))
            }
        }],
        "process_function": "process_nyu",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 365,
            "val": 91
        }
    },

    "berkeley_autolab_ur5": {
        "viewpoints": [{
            "lookat": [0.24398564,  0.23394822,  0.25454247],
            "distance": 0.36241041268887836,
            "azimuth": 140.5502567979282 - 180,
            "elevation": -42.125,
            "camera_fov": 57.82240163683314,
            "episodes": {
                "train": list(range(896)),
                "test": list(range(104))
            }
        }],
        "process_function": "process_berkeley_ur5",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "ur5e",
        "num_episodes": {
            "train": 896,
            "test": 104
        }
    },

    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.35624648, 0.18027023, 0.26440381],
            "distance": 1.0102697170162116,
            "azimuth": 91.74973347547972,
            "elevation": -3.5082622601279727,
            "camera_fov": 45,
            "episodes": {
                "train": list(range(150)),
            }
        }],
        "process_function": "process_ucsd_kitchen",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "use_joint_angles": True,
        "source_robot": "xarm7",
        "num_episodes": {
            "train": 150,
        },
    },

    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.16423974, 0.0605987, 0.13352152],
            "distance": 0.9653874337601709,
            "azimuth": 174.6463573393588,
            "elevation": -34.48271394183545,
            "camera_fov": 45,
            "episodes": {
                "train": list(range(92)),
                "val": list(range(10))
            }
        }],
        "process_function": "process_utokyo_xarm_pick_place",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "xarm7",
        "num_episodes": {
            "train": 92,
            "val": 10
        }
    },

    "kaist_nonprehensile_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.40113128, 0.1131025, -0.20421406],
            "distance": 0.9116127545512921,
            "azimuth": 139.6851649044909,
            "elevation": -37.03850699107953,
            "camera_fov": 43.0,
            "episodes": {
                "train": list(range(201)),
            }
        }],
        "process_function": "process_kaist_nonprehensile",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 201,
        },
    },

    "austin_buds_dataset_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.43482858, 0.08394756, -0.18341207],
            "distance": 1.1236742324339846,
            "azimuth": 179.10815957988817,
            "elevation": -60.54061958405549,
            "camera_fov": 50,
            "episodes": {
                "train": list(range(50)),
            }
        }],
        "process_function": "process_austin_buds",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 50,
        },
    },

    "austin_sailor_dataset_converted_externally_to_rlds": {
        "viewpoints": [{
            "lookat": [0.35071924, 0.39197068, -0.66741147],
            "distance": 1.570611132488425,
            "azimuth": 113.87333894803362,
            "elevation": -49.36992403626616,
            "camera_fov": 46,
            "episodes": {
                "train": list(range(240)),
            }
        }],
        "process_function": "process_austin_sailor",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 240,
        },
    },

    "utaustin_mutex": {
        "viewpoints": [{
            "lookat": [0.43803405, 0.04173897, 0.19695073],
            "distance": 0.6745413646576816,
            "azimuth": -37.29853605196638,
            "elevation": -32.75,
            "camera_fov": 60,
            "episodes": {
                "train": list(range(1500)),
            }
        }],
        "process_function": "process_austin_mutex",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 1500,
        },
    },

    "viola": {
        "viewpoints": [
            {
                "lookat": [0.52363341, 0.15771185, 0.05883797],
                "distance": 1.06776311996394,
                "azimuth": 120.93749999999997,
                "elevation": -32.79285699999997,
                "camera_fov": 36.0,
                "episodes": {
                    "train": [10, 100, 102, 110, 111, 116, 117, 120, 126, 129, 130, 132, 18, 20, 21, 23, 25, 28, 29, 3, 30, 34, 38, 39, 4, 43, 45, 52, 54, 55, 56, 63, 64, 67, 68, 74, 77, 79, 80, 84, 86, 9, 96, 97, 98],
                    "test": [3, 4, 7, 10, 12]
                }
            },
            {
                "lookat": [0.42379717, 0.0389163, -0.25703527],
                "distance": 1.3490181765881022,
                "azimuth": -179.59821428571428,
                "elevation": -46.453571285714276,
                "camera_fov": 36.0,
                "episodes": {
                    "train": [0, 1, 101, 103, 104, 105, 106, 107, 108, 109, 11, 112, 113, 114, 115, 118, 119, 12, 121, 122, 123, 124, 125, 127, 128, 13, 131, 133, 134, 14, 15, 16, 17, 19, 2, 22, 24, 26, 27, 31, 32, 33, 35, 36, 37, 40, 41, 42, 44, 46, 47, 48, 49, 5, 50, 51, 53, 57, 58, 59, 6, 60, 61, 62, 65, 66, 69, 7, 70, 71, 72, 73, 75, 76, 78, 8, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 99]
,
                    "test": [0, 1, 2, 5, 6, 8, 9, 11, 13, 14]
                }
            }
        ],
        "process_function": "process_viola",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 135,
            "test": 15
        }
    },

    "taco_play": {
            "viewpoints": [{
            "lookat": [0.38486548, -0.03720811,  0.37050499],
            "distance": 1.3067550966272043,
            "azimuth": 41.520000000000046,
            "elevation": -36.67999999999999,

            "episodes": {
                "train": list(range(3242)),
                "test": list(range(361))
            }
        }],
        "process_function": "process_taco_play",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "panda",
        "num_episodes": {
            "train": 3242,
            "test": 361
        }
    },

    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "viewpoints": [
            #top view
            {
                "lookat": [0.62342301, 0.02242862, 0.09345403],       
                "distance": 0.6417597084455157,
                "azimuth": 179.75,
                "elevation": -89.0,
                "camera_fov": 45,  
                "episodes": {
                    "train": [10, 101, 107, 117, 118, 123, 127, 129, 132, 134, 139, 149, 150, 153, 154, 157, 159, 164, 166, 167, 170, 171, 173, 174, 175, 176, 178, 181, 185, 186, 187, 19, 190, 193, 195, 196, 197, 200, 205, 213, 215, 216, 217, 218, 22, 220, 223, 224, 225, 232, 236, 237, 240, 245, 246, 25, 250, 251, 252, 256, 261, 264, 272, 274, 281, 287, 288, 289, 29, 291, 297, 302, 306, 309, 31, 312, 314, 315, 317, 318, 32, 320, 328, 331, 334, 337, 347, 348, 35, 354, 356, 358, 359, 363, 365, 366, 368, 369, 37, 371, 374, 377, 378, 379, 38, 389, 393, 394, 398, 402, 403, 410, 419, 42, 422, 426, 429, 43, 431, 434, 44, 441, 447, 448, 45, 450, 457, 458, 459, 464, 470, 473, 475, 48, 481, 482, 483, 485, 486, 487, 489, 49, 492, 496, 497, 504, 507, 510, 511, 513, 516, 518, 52, 520, 521, 525, 526, 527, 529, 534, 537, 540, 542, 544, 546, 547, 548, 552, 553, 555, 557, 56, 560, 562, 563, 565, 57, 570, 576, 578, 579, 58, 587, 588, 590, 594, 595, 596, 597, 6, 600, 606, 607, 610, 614, 618, 621, 626, 627, 630, 65, 67, 68, 69, 74, 80, 87, 89, 94, 95, 96, 98]
                }
            },

            #side view
            {
                "lookat": [0.57016495, 0.3104143, 0.22921072],       
                "distance": 0.37968397397383385,
                "azimuth": -91.0,
                "elevation": -43.25,
                "camera_fov": 45,   
                "episodes": {
                    "train": [0, 1, 100, 102, 103, 104, 105, 106, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 119, 12, 120, 121, 122, 124, 125, 126, 128, 13, 130, 131, 133, 135, 136, 137, 138, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 15, 151, 152, 155, 156, 158, 16, 160, 161, 162, 163, 165, 168, 169, 17, 172, 177, 179, 18, 180, 182, 183, 184, 188, 189, 191, 192, 194, 198, 199, 2, 20, 201, 202, 203, 204, 206, 207, 208, 209, 21, 210, 211, 212, 214, 219, 221, 222, 226, 227, 228, 229, 23, 230, 231, 233, 234, 235, 238, 239, 24, 241, 242, 243, 244, 247, 248, 249, 253, 254, 255, 257, 258, 259, 26, 260, 262, 263, 265, 266, 267, 268, 269, 27, 270, 271, 273, 275, 276, 277, 278, 279, 28, 280, 282, 283, 284, 285, 286, 290, 292, 293, 294, 295, 296, 298, 299, 3, 30, 300, 301, 303, 304, 305, 307, 308, 310, 311, 313, 316, 319, 321, 322, 323, 324, 325, 326, 327, 329, 33, 330, 332, 333, 335, 336, 338, 339, 34, 340, 341, 342, 343, 344, 345, 346, 349, 350, 351, 352, 353, 355, 357, 36, 360, 361, 362, 364, 367, 370, 372, 373, 375, 376, 380, 381, 382, 383, 384, 385, 386, 387, 388, 39, 390, 391, 392, 395, 396, 397, 399, 4, 40, 400, 401, 404, 405, 406, 407, 408, 409, 41, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 423, 424, 425, 427, 428, 430, 432, 433, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 449, 451, 452, 453, 454, 455, 456, 46, 460, 461, 462, 463, 465, 466, 467, 468, 469, 47, 471, 472, 474, 476, 477, 478, 479, 480, 484, 488, 490, 491, 493, 494, 495, 498, 499, 5, 50, 500, 501, 502, 503, 505, 506, 508, 509, 51, 512, 514, 515, 517, 519, 522, 523, 524, 528, 53, 530, 531, 532, 533, 535, 536, 538, 539, 54, 541, 543, 545, 549, 55, 550, 551, 554, 556, 558, 559, 561, 564, 566, 567, 568, 569, 571, 572, 573, 574, 575, 577, 580, 581, 582, 583, 584, 585, 586, 589, 59, 591, 592, 593, 598, 599, 60, 601, 602, 603, 604, 605, 608, 609, 61, 611, 612, 613, 615, 616, 617, 619, 62, 620, 622, 623, 624, 625, 628, 629, 63, 64, 66, 7, 70, 71, 72, 73, 75, 76, 77, 78, 79, 8, 81, 82, 83, 84, 85, 86, 88, 9, 90, 91, 92, 93, 97, 99]
                }
            }
        
        ],
        "process_function": "process_iamlab_cmu_pickup_insert",
        "out_path": "../videos",
        "extend_gripper": 0.08,
        "source_robot": "panda",
        "num_episodes": {
            "train": 631,
        }
    },

    "language_table": {
        "viewpoints": [{
            "lookat": [ 0.35058051, -0.01938161, -0.15156436],
            "distance": 0.668345794864296,
            "azimuth": -178.6875,
            "elevation": -50.59375,
            "episodes": {
                "train": list(range(442226)),
            }
        }],
        "process_function" : "process_language_table",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "source_robot": "xarm7",
        "num_episodes": {
            "train": 902,
            "test": 101
        }
    },

    "jaco_play": {
        "viewpoints": [{
            "lookat": [-0.13805259, -0.46769665,  0.01131256],
            "distance": 0.790969149319087,
            "azimuth": 88.5309361049106,
            "elevation": -44.80357142857134,
            "episodes": {
                "train": list(range(976)),
                "test": list(range(109))
            }
        }],
        "process_function" : "process_jaco_play",
        "out_path": "../videos",
        "extend_gripper": 0.0,
        "use_joint_angles": True,
        "source_robot": "jaco",
        "viewpoint_matching": False,
        "num_episodes": {
            "train": 976,
            "test": 101
        }
    },


    "fractal20220817_data": {
        "viewpoints": [
        {
            "lookat": [0.705072111870345, 0.013975478534784026, 0.6680251974054469],
            "distance": 0.7979868858827744,
            "azimuth": -2.4443359375,
            "elevation": -40.873046875,
            "camera_fov": 57
        },
        {
            "lookat": [0.7048949088644558, 0.02749863352837792, 0.6741001984235468],
            "distance": 0.792821361526563,
            "azimuth": 0.587890625,
            "elevation": -39.203125,
            "camera_fov": 57
        },
        {
            "lookat": [0.7052842721455245, -0.016368015238348046, 0.6339465158322075],
            "distance": 0.7954041237046687,
            "azimuth": -1.3896484375,
            "elevation": -42.630859375,
            "camera_fov": 57
        },
        {
            "lookat": [0.7024345226082027, -0.0028704309768615534, 0.6727576453286863],
            "distance": 0.7407394917247053,
            "azimuth": -1.51904296875,
            "elevation": -39.7607421875,
            "camera_fov": 57
        },
        {
            "lookat": [0.7053134200731075, 0.0005384909683360795, 0.7008562941856086],
            "distance": 0.7414789834494107,
            "azimuth": -2.7958984375,
            "elevation": -39.818359375,
            "camera_fov": 57
        },
        {
            "lookat": [0.70533450282253, 0.003704349859609352, 0.5796376127928865],
            "distance": 0.8599994388268911,
            "azimuth": -0.113037109375,
            "elevation": -45.603515625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6997674313001118, -0.020530853148594195, 0.6614478955343539],
            "distance": 0.7841997517837511,
            "azimuth": -9.605224609375,
            "elevation": -41.736328125,
            "camera_fov": 57
        },
        {
            "lookat": [0.7002655265592601, -0.010816362621324512, 0.5573805217762815],
            "distance": 0.85,
            "azimuth": 2.109375,
            "elevation": -49.21875,
            "camera_fov": 57
        },
        {
            "lookat": [0.6979446018967236, -0.005879223835473823, 0.6729901255903193],
            "distance": 0.7799865335834213,
            "azimuth": -1.167724609375,
            "elevation": -43.494140625,
            "camera_fov": 57
        },
        {
            "lookat": [0.69789022826663, -0.0025710913446387344, 0.6399018883029551],
            "distance": 0.7799865335834213,
            "azimuth": 0.941650390625,
            "elevation": -44.197265625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6979308146711577, -0.007549449747350338, 0.5637507722956394],
            "distance": 0.7799865335834213,
            "azimuth": -0.464599609375,
            "elevation": -47.009765625,
            "camera_fov": 57
        },
        {
            "lookat": [0.698585150958781, 0.025545750606431557, 0.6878787317455327],
            "distance": 0.7799865335834213,
            "azimuth": -1.167724609375,
            "elevation": -43.494140625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6977380589887675, 0.012347976290985843, 0.7292335562647291],
            "distance": 0.7799865335834213,
            "azimuth": -5.034912109375,
            "elevation": -41.736328125,
            "camera_fov": 57
        },
        {
            "lookat": [0.6981461138647066, 0.004006839260568541, 0.6030605443963426],
            "distance": 0.8326134820570754,
            "azimuth": -1.167724609375,
            "elevation": -43.494140625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6986173140485316, -0.02036985412694068, 0.6962989431759237],
            "distance": 0.7841997517837511,
            "azimuth": -1.870849609375,
            "elevation": -38.923828125,
            "camera_fov": 57
        },
        {
            "lookat": [0.6984640079625198, -0.02172122539325369, 0.6912207821428602],
            "distance": 0.7624446283289074,
            "azimuth": -1.519287109375,
            "elevation": -39.978515625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6472670402681677, -0.0326971252495982, 0.9442527585861564],
            "distance": 0.40898720590602416,
            "azimuth": -10.308349609375,
            "elevation": -72.125,
            "camera_fov": 57
        },
        {
            "lookat": [0.6558801257686773, -0.001365699335457367, 0.6433031522284819],
            "distance": 0.8144390731619926,
            "azimuth": -15.230224609375,
            "elevation": -60.017578125,
            "camera_fov": 57
        },
        {
            "lookat": [0.6608054839958618, 0.016701091684091805, 0.7963179328599144],
            "distance": 0.8082898446741533,
            "azimuth": -21.558349609375,
            "elevation": -49.119140625,
            "camera_fov": 57
        },
        {
            "lookat": [0.6436299634139799, -0.042107918018819, 0.7080959925655009],
            "distance": 0.6828806963949621,
            "azimuth": -8.550537109375,
            "elevation": -42.791015625,
            "camera_fov": 57
        }
    ],
        "process_function": "process_fractal20220817_data",
        "out_path": "../videos",
        "extend_gripper": 0,
        "source_robot": "google_robot"
    },

    "bridge": {
        "viewpoints": bridge_viewpoints,
        "process_function": "process_bridge",
        "out_path": "../videos",
        "extend_gripper": 0,
        "num_episodes": 10,
        "use_joint_angles": False,
        "viewpoint_matching": True,
        "source_robot": "widowX",
    }
}


_DEFAULT_PROCESSING_MODULE = "core.processing_utils"


@lru_cache(None)
def _load_module(module_path: str):
    return importlib.import_module(module_path)


def resolve_process_function(dataset_name: str):
    dataset_cfg = config.get(dataset_name)
    if dataset_cfg is None:
        raise KeyError(f"Unknown dataset {dataset_name}")

    func_name = dataset_cfg.get("process_function")
    if func_name is None:
        raise KeyError(f"Dataset {dataset_name} is missing 'process_function'")

    module_path = dataset_cfg.get("process_module", _DEFAULT_PROCESSING_MODULE)
    module = _load_module(module_path)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"Module {module_path} does not define '{func_name}'") from exc
