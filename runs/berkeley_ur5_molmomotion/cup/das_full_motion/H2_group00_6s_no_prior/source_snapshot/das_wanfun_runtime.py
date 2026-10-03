"""Single GPU compatibility adapter for the pinned official DaS Wanfun source.

The upstream commit omits videox_fun/dist although model modules import it.
Only unused distributed interfaces are supplied. Distributed execution fails
explicitly. Model code, weights, attention, VAE, and offload are unchanged.
"""
import ast
import importlib
import os
import sys
import types
from pathlib import Path


def unavailable(*args, **kwargs):
    raise RuntimeError('Distributed execution is unavailable in this single GPU adapter')


def setup(das_root):
    sys.path.insert(0, str(das_root))
    if not (Path(das_root)/'videox_fun/dist').exists():
        dist = types.ModuleType('videox_fun.dist')
        dist.__path__ = []
        dist.get_sequence_parallel_rank = lambda: 0
        dist.get_sequence_parallel_world_size = lambda: 1
        dist.get_sp_group = unavailable
        dist.xFuserLongContextAttention = unavailable
        # Upstream decorator is for distributed spatial VAE execution;
        # the original decode function is retained on one GPU.
        dist.parallel_magvit_vae = lambda *args, **kwargs: lambda function: function
        sys.modules[dist.__name__] = dist
        wan = types.ModuleType('videox_fun.dist.wan_xfuser')
        wan.usp_attn_forward = unavailable
        sys.modules[wan.__name__] = wan
    # Package __init__ eagerly imports unrelated CogVideoX backends and their
    # missing distributed code. Load only the unchanged Wan model modules.
    if 'videox_fun.models' not in sys.modules:
        models = types.ModuleType('videox_fun.models')
        models.__path__ = [str(Path(das_root)/'videox_fun/models')]
        sys.modules[models.__name__] = models
        from transformers import AutoTokenizer
        models.AutoTokenizer = AutoTokenizer
        for module_name, class_name in [('wan_transformer3d','WanTransformer3DModel'),
                                        ('wan_vae','AutoencoderKLWan'),
                                        ('wan_text_encoder','WanT5EncoderModel'),
                                        ('wan_image_encoder','CLIPModel')]:
            module = importlib.import_module('videox_fun.models.'+module_name)
            setattr(models,class_name,getattr(module,class_name))
    if 'videox_fun.pipeline' not in sys.modules:
        pipeline = types.ModuleType('videox_fun.pipeline')
        pipeline.__path__ = [str(Path(das_root)/'videox_fun/pipeline')]
        sys.modules[pipeline.__name__] = pipeline
        module=importlib.import_module('videox_fun.pipeline.pipeline_wan_fun_control')
        pipeline.WanFunControlPipeline = module.WanFunControlPipeline


def load_official_infer(das_root):
    """Extract the official method without loading unused MoGe/trackers/Flux.

Only three unused local import statements are removed. The method body is
otherwise compiled directly from the pinned upstream file.
"""
    import torch
    import numpy as np
    from PIL import Image
    setup(das_root)
    source = Path(das_root)/'models/pipelines_wanfun.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='DiffusionAsShaderPipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_infer_wanfun_ctrl')
    method.body = [n for n in method.body if not (isinstance(n, ast.ImportFrom) and
                   (n.module=='models.cogvideox_tracking' or
                    n.module=='transformers' and any(a.name=='T5EncoderModel' for a in n.names) or
                    n.module=='diffusers' and any(a.name=='AutoencoderKLCogVideoX' for a in n.names)))]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {'torch':torch, 'np':np, 'Image':Image, 'os':os}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['_infer_wanfun_ctrl']
