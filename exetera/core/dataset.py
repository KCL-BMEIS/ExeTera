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
from exetera.core.abstract_types import Dataset
from exetera.core import dataframe as edf


class HDF5Dataset(Dataset):

    def __init__(self, session, dataset_path, mode, name):
        self._session = session
        self.file = h5py.File(dataset_path, mode)
        self.dataframes = dict()
        for subgrp in self.file.keys():
            hdf = edf.HDF5DataFrame(self,subgrp,h5group=self.file[subgrp])
            self.dataframes[subgrp]=hdf

    @property
    def session(self):
        return self._session


    def close(self):
        self.file.close()


    def create_dataframe(self, name):
        """
        Create a group object in HDF5 file and a Exetera dataframe in memory.

        :param name: the name of the group and dataframe
        :return: a dataframe object
        """
        self.file.create_group(name)
        dataframe = edf.HDF5DataFrame(self, name)
        self.dataframes[name] = dataframe
        return dataframe


    def add(self, dataframe, name=None):
        """
        Add an existing dataframe to this dataset, write the existing group
        attributes and HDF5 datasets to this dataset.

        :param dataframe: the dataframe to copy to this dataset
        :param name: optional- change the dataframe name
        """
        dname = dataframe.name if name is None else name
        self.file.copy(dataframe.dataset.file[dataframe.name], self.file, name=dname)
        df = edf.HDF5DataFrame(self, dname, h5group=self.file[dname])
        self.dataframes[dname] = df


    def __contains__(self, name):
        return self.dataframes.__contains__(name)


    def contains_dataframe(self, dataframe):
        """
        Check if a dataframe is contained in this dataset by the dataframe object itself.

        :param dataframe: the dataframe object to check
        :return: Ture or False if the dataframe is contained
        """
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field must be a DataFrame object")
        else:
            for v in self.dataframes.values():
                if id(dataframe) == id(v):
                    return True
            return False


    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not self.__contains__(name):
            raise ValueError("Can not find the name from this dataset.")
        else:
            return self.dataframes[name]


    def get_dataframe(self, name):
        self.__getitem__(name)


    def get_name(self, dataframe):
        """
        Get the name of the dataframe in this dataset.
        """
        if not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field argument must be a DataFrame object.")
        for name, v in self.fields.items():
            if id(dataframe) == id(v):
                return name
                break
        return None

    def __setitem__(self, name, dataframe):
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not isinstance(dataframe, edf.DataFrame):
            raise TypeError("The field must be a DataFrame object.")
        else:
            if self.dataframes.__contains__(name):
                self.__delitem__(name)
            return self.add(dataframe,name)


    def __delitem__(self, name):
        if not self.__contains__(name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self.dataframes[name]
            del self.file[name]
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


    def list(self):
        return tuple(n for n in self.dataframes.keys())


    def keys(self):
        return self.dataframes.keys()


    def values(self):
        return self.dataframes.values()


    def items(self):
        return self.dataframes.items()


    def __iter__(self):
        return iter(self.dataframes)


    def __next__(self):
        return next(self.dataframes)


    def __len__(self):
        return len(self.dataframes)
