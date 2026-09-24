"""FP32 inference utilities."""
from pathlib import Path
import copy
import random

import numpy as np
import torch
from mmcv import Config
from mmrotate.models import build_detector

ROOT = Path(__file__).resolve().parent
MODALITIES = ('sar', 'rgb', 'ir')


def setup(seed=0):
    """Set the seed and FP32 backend settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def load_config(path):
    """Load the model architecture and modality preprocessing."""
    return Config.fromfile(str(Path(path).resolve()))


def load_model(cfg, checkpoint, device='cuda:0'):
    """Load model weights."""
    config = copy.deepcopy(cfg.model)
    config.pretrained = None
    config.backbone.init_cfg = None
    model = build_detector(config)
    payload = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.CLASSES = payload.get('meta', {}).get('CLASSES')
    model.cfg = cfg
    return model.float().to(device).eval()
