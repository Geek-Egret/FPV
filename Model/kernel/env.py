import numpy as np
import torch 
import math

import kernel.util as util

class env:
    def __init__(self, solver):
        self._solver = solver
    
    """
        添加圆柱
        R:半径:m
        H:高度
    """
    