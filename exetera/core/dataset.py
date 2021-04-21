# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py

from exetera.core.abstract_types import DataFrame, Dataset
from exetera.core import dataframe as edf


class HDF5Dataset(Dataset):
    """
    Dataset is the means which which you interact with an ExeTera datastore. These are created
    and loaded through `Session.open_dataset`, rather than being constructed directly.

    Datasets are composed of one or more DataFrame objects and the means by which DataFrames
    are interacted with.

    For a detailed explanation of Dataset along with examples of its use, please refer to the
    wiki documentation at
    https://github.com/KCL-BMEIS/ExeTera/wiki/Dataset-API
    """

    def __init__(self, session, dataset_path, mode, name):
        """
        Create a HDF5Dataset instance that contains dataframes. The dataframes are represented in a dict() with the
        name(str) as a key. The construction should always be called by Session.open_dataset() otherwise the instance
        is not included in Session.datasets. If the HDF5 datafile contains group, the content in loaded into dataframes.

        :param session: The session instance to include this dataset to.
        :param dataset_path: The path of HDF5 file.
        :param mode: the mode in which the dataset should be opened. This is one of "r", "r+" or "w".
        :param name: the name that is associated with this dataset. This can be used to retrieve the dataset when
        calling :py:meth:`~session.Session.get_dataset`.
        :return: A HDF5Dataset instance.
        """
        self.name = name
        self._session = session
        self._file = h5py.File(dataset_path, mode)
        self._dataframes = dict()

        for group in self._file.keys():
            if group not in ('trash',):
                h5group = self._file[group]
                dataframe = edf.HDF5DataFrame(self, group, h5group=h5group)
                self._dataframes[group] = dataframe

    @property
    def session(self):
        """
        The session property interface.
        
        :return: The _session instance.
        """
        return self._session

    def create_dataframe(self, name, dataframe: DataFrame = None):
        """
        Create a group object in HDF5 file and a Exetera dataframe in memory.

        :param name: name of the dataframe, or the group name in HDF5
        :param dataframe: optional - copy an existing dataframe
        :return: a dataframe object
        """
        if dataframe is not None:
            if not isinstance(dataframe, DataFrame):
                raise ValueError("If set, 'dataframe' must be of type DataFrame "
                                 "but is of type {}".format(type(dataframe)))

        self._file.create_group(name)
        h5group = self._file[name]
        _dataframe = edf.HDF5DataFrame(self, name, h5group)
        if dataframe is not None:
            for k, v in dataframe.items():
                f = v.create_like(_dataframe, k)
                if f.indexed:
                    f.indices.write(v.indices[:])
                    f.values.write(v.values[:])
                else:
                    f.data.write(v.data[:])

        self._dataframes[name] = _dataframe
        return _dataframe

    def close(self):
        """Close the HDF5 file operations."""
        self._file.close()

    def copy(self, dataframe, name):
        """
        Add an existing dataframe (from other dataset) to this dataset, write the existing group
        attributes and HDF5 datasets to this dataset.

        :param dataframe: the dataframe to copy to this dataset.
        :param name: optional- change the dataframe name.
        :return: None if the operation is successful; otherwise throw Error.
        """
        copy(dataframe, self, name)
        # dname = dataframe.name
        # self._file.create_group(dname)
        # h5group = self._file[dname]
        # _dataframe = edf.HDF5DataFrame(self, dname, h5group)
        # for k, v in dataframe.items():
        #     f = v.create_like(_dataframe, k)
        #     if f.indexed:
        #         f.indices.write(v.indices[:])
        #         f.values.write(v.values[:])
        #     else:
        #         f.data.write(v.data[:])
        # self._dataframes[dname] = _dataframe

    def __contains__(self, name: str):
        """
        Check if the name exists in this dataset.

        :param name: Name of the dataframe to check.
        :return: Boolean if the name exists.
        """
        return name in self._dataframes

    def contains_dataframe(self, dataframe: DataFrame):
        """
        Check if a dataframe is contained in this dataset by the dataframe object itself.

        :param dataframe: the dataframe object to check
        :return: Ture or False if the dataframe is contained
        """
        if not isinstance(dataframe, DataFrame):
            raise TypeError("The field must be a DataFrame object")
        else:
            for v in self._dataframes.values():
                if id(dataframe) == id(v):
                    return True
            return False

    def __getitem__(self, name: str):
        """
        Get the dataframe by dataset[dataframe_name].

        :param name: The name of the dataframe to get.
        """
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not self.__contains__(name):
            raise ValueError("Can not find the name from this dataset.")
        else:
            return self._dataframes[name]

    def get_dataframe(self, name: str):
        """
        Get the dataframe by dataset.get_dataframe(dataframe_name).

        :param name: The name of the dataframe.
        :return: The dataframe or throw Error if the name is not existed in this dataset.
        """
        self.__getitem__(name)

    def __setitem__(self, name: str, dataframe: DataFrame):
        """
        Add an existing dataframe (from other dataset) to this dataset, the existing dataframe can from:
        1) this dataset, so perform a 'rename' operation, or;
        2) another dataset, so perform a copy operation

        :param name: The name of the dataframe to store in this dataset.
        :param dataframe: The dataframe instance to store in this dataset.
        :return: None if the operation is successful; otherwise throw Error.
        """
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field must be a DataFrame object.")

        if dataframe.dataset == self:
            # rename a dataframe
            del self._dataframes[dataframe.name]
            dataframe.name = name
            self._file.move(dataframe.h5group.name, name)
        else:
            # new dataframe from another dataset
            copy(dataframe, self, name)

    def __delitem__(self, name: str):
        """
        Delete a dataframe by del dataset[name].
        :param name: The name of dataframe to delete.
        :return: Boolean if the dataframe is deleted.
        """
        if not self.__contains__(name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self._dataframes[name]
            del self._file[name]
            return True

    def delete_dataframe(self, dataframe: DataFrame):
        """
        Remove dataframe from this dataset by the dataframe object.
        :param dataframe: The dataframe instance to delete.
        :return: Boolean if the dataframe is deleted.
        """
        #name = self.get_name(dataframe)
        name = dataframe.name
        if name is None:
            raise ValueError("This dataframe does not contain the field to delete.")
        else:
            self.__delitem__(name)

    def drop(self,
             name: str):
        del self._dataframes[name]
        del self._file[name]

    def keys(self):
        """Return all dataframe names in this dataset."""
        return self._dataframes.keys()

    def values(self):
        """Return all dataframe instance in this dataset."""
        return self._dataframes.values()

    def items(self):
        """Return the (name, dataframe) tuple in this dataset."""
        return self._dataframes.items()

    def __iter__(self):
        """Iteration through the dataframes stored in this dataset."""
        return iter(self._dataframes)

    def __next__(self):
        """Next dataframe for iteration through dataframes stored."""
        return next(self._dataframes)

    def __len__(self):
        """Return the number of dataframes stored in this dataset."""
        return len(self._dataframes)


def copy(dataframe: DataFrame, dataset: Dataset, name: str):
    """
    Copy dataframe to another dataset via HDF5DataFrame.copy(ds1['df1'], ds2, 'df1'])

    :param dataframe: The dataframe to copy.
    :param dataset: The destination dataset.
    :param name: The name of dataframe in destination dataset.
    """
    if name in dataset:
        raise ValueError("A dataframe with the the name {} already exists in the "
                         "destination dataset".format(name))

    _dataframe = dataset.create_dataframe(name)

    for k, v in dataframe.items():
        f = v.create_like(_dataframe, k)
        if f.indexed:
            f.indices.write(v.indices[:])
            f.values.write(v.values[:])
        else:
            f.data.write(v.data[:])

    dataset._dataframes[name] = _dataframe


def move(dataframe: DataFrame, dataset: Dataset, name:str):
    """
    Move a dataframe to another dataset via HDF5DataFrame.move(ds1['df1'], ds2, 'df1']).
    If move within the same dataset, e.g. HDF5DataFrame.move(ds1['df1'], ds1, 'df2']), function as a rename for both
    dataframe and HDF5Group. However, to

    :param dataframe: The dataframe to copy.
    :param dataset: The destination dataset.
    :param name: The name of dataframe in destination dataset.
    """
    copy(dataframe, dataset, name)
    dataframe.dataset.drop(dataframe.name)
