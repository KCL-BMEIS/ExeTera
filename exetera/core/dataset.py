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
from exetera.core.abstract_types import Dataset,DataFrame
from exetera.core import dataframe as edf


class HDF5Dataset(Dataset):

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
        for subgrp in self._file.keys():
            self.create_dataframe(subgrp, h5group=self._file[subgrp])

    @property
    def session(self):
        """
        The session property interface.

        :return: The _session instance.
        """
        return self._session

    def close(self):
        """Close the HDF5 file operations."""
        self._file.close()

    def create_dataframe(self, name, dataframe: dict = None, h5group: h5py.Group = None):
        """
        Create a group object in HDF5 file and a Exetera dataframe in memory.

        :param name: name of the dataframe, or the group name in HDF5
        :param dataframe: optional - replicate data from another dictionary
        :param h5group: optional - acquire data from h5group object directly, the h5group needs to have a
                        h5group<-group-dataset structure, the group has a 'fieldtype' attribute
                         and the dataset is named 'values'.
        :return: a dataframe object
        """
        if h5group is None:
            self._file.create_group(name)
            h5group = self._file[name]
        dataframe = edf.HDF5DataFrame(self, name, h5group, dataframe)
        self._dataframes[name] = dataframe
        return dataframe

    def add(self, dataframe, name=None):
        """
        Add an existing dataframe (from other dataset) to this dataset, write the existing group
        attributes and HDF5 datasets to this dataset.

        :param dataframe: the dataframe to copy to this dataset.
        :param name: optional- change the dataframe name.
        :return: None if the operation is successful; otherwise throw Error.
        """
        dname = dataframe.name if name is None else name
        self._file.copy(dataframe.h5group, self._file, name=dname)
        df = edf.HDF5DataFrame(self, dname, h5group=self._file[dname])
        self._dataframes[dname] = df

    def __contains__(self, name: str):
        """
        Check if the name exists in this dataset.

        :param name: Name of the dataframe to check.
        :return: Boolean if the name exists.
        """
        return self._dataframes.__contains__(name)

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

    def get_name(self, dataframe: DataFrame):
        """
        If the dataframe exist in this dataset, return the name; otherwise return None.

        :param dataframe: The dataframe instance to find the name.
        :return: name (str) of the dataframe or None if dataframe not found in this dataset.
        """
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field argument must be a DataFrame object.")
        for name, v in self._dataframes.items():
            if id(dataframe) == id(v):
                return name
        return None

    def __setitem__(self, name: str, dataframe: DataFrame):
        """
        Add an existing dataframe (from other dataset) to this dataset, the existing dataframe can from:
        1) this dataset, so perform a 'rename' operation, or;
        2) another dataset, so perform an 'add' or 'replace' operation

        :param name: The name of the dataframe to store in this dataset.
        :param dataframe: The dataframe instance to store in this dataset.
        :return: None if the operation is successful; otherwise throw Error.
        """
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field must be a DataFrame object.")
        else:
            if dataframe.dataset == self:  # rename a dataframe
                return self._file.move(dataframe.name, name)
            else:  # new dataframe from another dataset
                if self._dataframes.__contains__(name):
                    self.__delitem__(name)
                return self.add(dataframe, name)

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
        name = self.get_name(dataframe)
        if name is None:
            raise ValueError("This dataframe does not contain the field to delete.")
        else:
            self.__delitem__(name)

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
