from __future__ import annotations

import os
import random
import logging

_logger = logging.getLogger(__name__)

def setup_seed(seed: int, torch_deterministic: bool=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        _logger.debug(f"numpy not installed, skip setup seed")
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch_deterministic:
            # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_deterministic(True)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except ImportError:
        _logger.debug(f"torch not installed, skip setup seed")
        
    return seed