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

class Dataset():
    """
    DataSet is a container of dataframes
    """
    def __init__(self,file_path,name):
        pass

    def close(self):
        pass

    def add(self, field, name=None):
        pass

    def __contains__(self, name):
        pass

    def contains_dataframe(self, dataframe):
        pass

    def __getitem__(self, name):
        pass

    def get_dataframe(self, name):
        pass

    def get_name(self, dataframe):
        pass

    def __setitem__(self, name, dataframe):
        pass

    def __delitem__(self, name):
        pass

    def delete_dataframe(self, dataframe):
        pass

    def list(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __len__(self):
        pass

import h5py
from exetera.core import dataframe as edf
class HDF5Dataset(Dataset):

    def __init__(self, dataset_path, mode, name):
        self.file = h5py.File(dataset_path, mode)
        self.dataframes = dict()

    def close(self):
        self.file.close()

    def create_dataframe(self,name):
        """
        Create a group object in HDF5 file and a Exetera dataframe in memory.

        :param name: the name of the group and dataframe
        :return: a dataframe object
        """
        self.file.create_group(name)
        dataframe = edf.DataFrame(name,self)
        self.dataframes[name]=dataframe
        return dataframe


    def add(self, dataframe, name=None):
        """
        Add an existing dataframe to this dataset, write the existing group
        attributes and HDF5 datasets to this dataset.

        :param dataframe: the dataframe to copy to this dataset
        :param name: optional- change the dataframe name
        """
        dname = dataframe if name is None else name
        self.file.copy(dataframe.dataset[dataframe.name],self.file,name=dname)
        df = edf.DataFrame(dname,self,h5group=self.file[dname])
        self.dataframes[dname]=df


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
                    break
            return False

    def __getitem__(self, name):
        if not isinstance(name,str):
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
            self.dataframes[name] = dataframe
            return True

    def __delitem__(self, name):
        if not self.__contains__(name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self.dataframes[name]
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
        return self.file.keys()

    def values(self):
        return self.file.values()

    def items(self):
        return self.file.items()

    def __iter__(self):
        return iter(self.dataframes)

    def __next__(self):
        return next(self.dataframes)

    def __len__(self):
        return len(self.dataframes)
