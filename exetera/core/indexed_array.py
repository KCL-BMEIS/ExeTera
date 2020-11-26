import numpy as np
from exetera.core import operations as ops

class IndexedArray:

    def __init__(self):
        self._indices = None
        self._values = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ops.apply_slice_to_index_values(index, self._indices, self._values)
        if isinstance(index, np.ndarray):
            if index.dtype == np.bool:
                return ops.apply_filter_to_index_values(index, self._indices, self._values)
            if np.issubdtype(index.dtype, np.integer):
                return ops.apply_indices_to_index_values(index, self._indices, self._values)

