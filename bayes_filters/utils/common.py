# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from typing import Union, Optional, Any, Callable, Tuple, List


__all__ = [
    "ArrayLike",
    "ArrayLike_Float",
    "ArrayLike_Int",
]


ArrayLike = Union[list,tuple,np.ndarray]
ArrayLike_Float = Union[List[float],Tuple[float],np.ndarray]
ArrayLike_Int = Union[List[int],Tuple[int],np.ndarray]
