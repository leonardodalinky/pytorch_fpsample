import importlib.machinery
import os.path as osp

import torch

from .fps import sample

__VERSION__ = "0.1.0"

torch.ops.load_library(
    importlib.machinery.PathFinder().find_spec(f"_core", [osp.dirname(__file__)]).origin
)
