import numpy as np

from exetera.core.dataframe import HDF5DataFrame
from exetera.core.fields import MemoryField


class ViewerMask:
    """
    Fetch dataframe through the viewer mask will only allow rows/columns listed in the mask.
    """
    def __init__(self, indexlist, columnlist):
        """
        Initialise a mask.

        :param indexlist: An ndarray of integers, indicating the index of elements to show.
        :param columnlist: An ndarray of strings, indicating the name of column to show.
        """
        self._index = indexlist
        self._column = columnlist

    @property
    def index(self):
        return self._index

    @property
    def column(self):
        return self._column

    def __and__(self, other):
        """
        Create a new mask by the intersected elements in both masks.

        Example::

            self: index [1, 2, 3, 4], column ['a', 'b']
            other: index [1, 2, 5, 6], column ['a', 'c']
            return: index [1, 2], column ['a']

        :param other: Another mask.
        :return: A new mask with intersected elements in both masks.
        """
        index = np.intersect1d(self.index, other.index)
        column = np.intersect1d(self.column, other.column)
        return ViewerMask(index, column)

    def __or__(self, other):
        """
        Create a new mask by the union elements in both masks.

        Example::

            self: index [1, 2, 3, 4], column ['a', 'b']
            other: index [1, 2, 5, 6], column ['a', 'c']
            return: index [1, 2, 3, 4, 5, 6], column ['a', 'b', 'c']

        :param other: Another mask.
        :return: A new mask with elements in both masks.
        """
        index = np.union1d(self.index, other.index)
        column = np.union1d(self.column, other.column)
        return ViewerMask(index, column)


class Viewer:
    """
    A viewer is a projected filter of a dataframe
    """
    def __init__(self, df, mask=None):
        if isinstance(df, HDF5DataFrame):
            self.df = df
            self.storage = df.h5group
            if mask is not None and isinstance(mask, ViewerMask):
                self._mask = mask
            else:
                self._mask = None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, msk):
        if isinstance(msk, ViewerMask):
            self._mask = msk

    def __getitem__(self, item):  # aka apply filter?
        if isinstance(item, str):  # df.loc['cobra'] filter on fields, return a field
            if item not in self._mask.column:
                raise ValueError("{} is not listed in the ViewerMask.".format(item))
            else:
                return self.df[item].data[self._mask.index]  # return data instread of field
        elif isinstance(item, slice):
            print("slice")
        elif isinstance(item, list):  # df.loc[[True, True, False]] filter on index, return a df
            pass
        elif isinstance(item, MemoryField):  # df.loc[df['shield'] > 35] filter on index, return a df
            pass
        elif isinstance(item, tuple):  # df.loc[_index , _column]
            if isinstance(item[0], slice):  # df.loc[:, ] = 30 filter
                pass
            elif isinstance(item[0], str):  # df.loc['abc',]
                pass

    def __setitem__(self, key, value):
        raise NotImplementedError("Please update field values though dataframe instead of viewer.")
