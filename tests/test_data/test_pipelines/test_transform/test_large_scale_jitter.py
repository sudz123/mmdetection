# Copyright (c) OpenMMLab. All rights reserved.

import copy

import mmcv
import numpy as np
from mmcv.utils import build_from_cfg
from numpy.testing import assert_array_equal

from mmdet.datasets.builder import PIPELINES
from .utils import construct_toy_data

