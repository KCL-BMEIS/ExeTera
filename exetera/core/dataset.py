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

    def __init__(self, session, dataset_path, mode, name):
        self.name = name
        self._session = session
        self._file = h5py.File(dataset_path, mode)
        self._dataframes = dict()

        for group in self._file.keys():
            h5group = self._file[group]
            dataframe = edf.HDF5DataFrame(self, group, h5group=h5group)
            self._dataframes[group] = dataframe

    @property
    def session(self):
        return self._session

    def close(self):
        self._file.close()

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

    def add(self, dataframe, name=None):
        """
        Add an existing dataframe (from other dataset) to this dataset, write the existing group
        attributes and HDF5 datasets to this dataset.

        :param dataframe: the dataframe to copy to this dataset
        :param name: optional- change the dataframe name
        """
        dname = dataframe.name if name is None else name
        self._file.copy(dataframe._h5group, self._file, name=dname)
        df = edf.HDF5DataFrame(self, dname, h5group=self._file[dname])
        self._dataframes[dname] = df

    def __contains__(self, name):
        return name in self._dataframes

    def contains_dataframe(self, dataframe):
        """
        Check if a dataframe is contained in this dataset by the dataframe object itself.

        :param dataframe: the dataframe object to check
        :return: Ture or False if the dataframe is contained
        """
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field must be a DataFrame object")
        else:
            for v in self._dataframes.values():
                if id(dataframe) == id(v):
                    return True
            return False

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not self.__contains__(name):
            raise ValueError("Can not find the name from this dataset.")
        else:
            return self._dataframes[name]

    def get_dataframe(self, name):
        self.__getitem__(name)

    def get_name(self, dataframe):
        """
        Get the name of the dataframe in this dataset.
        """
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field argument must be a DataFrame object.")
        for name, v in self._dataframes.items():
            if id(dataframe) == id(v):
                return name
        return None

    def __setitem__(self, name, dataframe):
        """
        Add an existing dataframe (from other dataset) to this dataset, the existing dataframe can from:
        1) this dataset, so perform a 'rename' operation, or;
        2) another dataset, so perform an 'add' or 'replace' operation
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

    def __delitem__(self, name):
        if not self.__contains__(name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self._dataframes[name]
            del self._file[name]
            return True

    def delete_dataframe(self, dataframe):
        """
        Remove dataframe from this dataset by dataframe object.
        """
        name = self.get_name(dataframe)
        if name is None:
            raise ValueError("This dataframe does not contain the field to delete.")
        else:
            self.__delitem__(name)

    def keys(self):
        return self._dataframes.keys()

    def values(self):
        return self._dataframes.values()

    def items(self):
        return self._dataframes.items()

    def __iter__(self):
        return iter(self._dataframes)

    def __next__(self):
        return next(self._dataframes)

    def __len__(self):
        return len(self._dataframes)
