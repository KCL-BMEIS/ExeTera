from exetera.core import fields as fld
from datetime import  datetime,timezone
import h5py

class DataFrame():
    """
    DataFrame is a table of data that contains a list of Fields (columns)
    """
    def __init__(self, name, dataset,data=None,h5group:h5py.Group=None):
        """
        Create a Dataframe object.

        :param name: name of the dataframe, or the group name in HDF5
        :param dataset: a dataset object, where this dataframe belongs to
        :param dataframe: optional - replicate data from another dictionary
        :param h5group: optional - acquire data from h5group object directly, the h5group needs to have a
                        h5group<-group-dataset structure, the group has a 'fieldtype' attribute
                         and the dataset is named 'values'.
        """
        self.fields = dict()
        self.name = name
        self.dataset = dataset

        if data is not None:
            if isinstance(data,dict) and isinstance(list(data.items())[0][0],str) and isinstance(list(data.items())[0][1], fld.Field) :
                self.fields=data
        elif h5group is not None and isinstance(h5group,h5py.Group):
            fieldtype_map = {
                'indexedstring': fld.IndexedStringField,
                'fixedstring': fld.FixedStringField,
                'categorical': fld.CategoricalField,
                'boolean': fld.NumericField,
                'numeric': fld.NumericField,
                'datetime': fld.TimestampField,
                'date': fld.TimestampField,
                'timestamp': fld.TimestampField
            }
            for subg in h5group.keys():
                fieldtype = h5group[subg].attrs['fieldtype'].split(',')[0]
                self.fields[subg] = fieldtype_map[fieldtype](self, h5group[subg])
                print(" ")

    def add(self,field,name=None):
        if name is not None:
            if not isinstance(name,str):
                raise TypeError("The name must be a str object.")
            else:
                self.fields[name]=field
        self.fields[field.name[field.name.index('/',1)+1:]]=field #note the name has '/' for hdf5 object

    def create_group(self,name):
        """
        Create a group object in HDF5 file for field to use.

        :param name: the name of the group and field
        :return: a hdf5 group object
        """
        self.dataset.file.create_group("/"+self.name+"/"+name)
        return self.dataset.file["/"+self.name+"/"+name]


    def create_numeric(self, session, name, nformat, timestamp=None, chunksize=None):
        fld.numeric_field_constructor(session, self, name, nformat, timestamp, chunksize)
        field=fld.NumericField(session, self.dataset.file["/"+self.name+"/"+name], write_enabled=True)
        self.fields[name]=field
        return self.fields[name]

    def create_indexed_string(self, session, name, timestamp=None, chunksize=None):
        fld.indexed_string_field_constructor(session, self, name, timestamp, chunksize)
        field= fld.IndexedStringField(session, self.dataset.file["/"+self.name+"/"+name], write_enabled=True)
        self.fields[name] = field
        return self.fields[name]

    def create_fixed_string(self, session, name, length, timestamp=None, chunksize=None):
        fld.fixed_string_field_constructor(session, self, name, length, timestamp, chunksize)
        field= fld.FixedStringField(session, self.dataset.file["/"+self.name+"/"+name], write_enabled=True)
        self.fields[name] = field
        return self.fields[name]

    def create_categorical(self, session, name, nformat, key,
                           timestamp=None, chunksize=None):
        fld.categorical_field_constructor(session, self, name, nformat, key,
                                          timestamp, chunksize)
        field= fld.CategoricalField(session, self.dataset.file["/"+self.name+"/"+name], write_enabled=True)
        self.fields[name] = field
        return self.fields[name]

    def create_timestamp(self, session, name, timestamp=None, chunksize=None):
        fld.timestamp_field_constructor(session, self, name, timestamp, chunksize)
        field= fld.TimestampField(session, self.dataset.file["/"+self.name+"/"+name], write_enabled=True)
        self.fields[name] = field
        return self.fields[name]


    def __contains__(self, name):
        """
        check if dataframe contains a field, by the field name
        name: the name of the field to check,return a bool
        """
        if not isinstance(name,str):
            raise TypeError("The name must be a str object.")
        else:
            return self.fields.__contains__(name)

    def contains_field(self,field):
        """
        check if dataframe contains a field by the field object
        field: the filed object to check, return a tuple(bool,str). The str is the name stored in dataframe.
        """
        if not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object")
        else:
            for v in self.fields.values():
                if id(field) == id(v):
                    return True
                    break
            return False

    def __getitem__(self, name):
        if not isinstance(name,str):
            raise TypeError("The name must be a str object.")
        elif not self.__contains__(name):
            raise ValueError("Can not find the name from this dataframe.")
        else:
            return self.fields[name]

    def get_field(self,name):
        return self.__getitem__(name)

    def get_name(self,field):
        """
        Get the name of the field in dataframe.
        """
        if not isinstance(field,fld.Field):
            raise TypeError("The field argument must be a Field object.")
        for name,v in self.fields.items():
            if id(field) == id(v):
                return name
                break
        return None

    def __setitem__(self, name, field):
        if not isinstance(name,str):
            raise TypeError("The name must be a str object.")
        elif not isinstance(field,fld.Field):
            raise TypeError("The field must be a Field object.")
        else:
            self.fields[name]=field
            return True

    def __delitem__(self, name):
        if not self.__contains__(name=name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self.fields[name]
            return True

    def delete_field(self,field):
        """
        Remove field from dataframe by field
        """
        name = self.get_name(field)
        if name is None:
            raise ValueError("This dataframe does not contain the field to delete.")
        else:
            self.__delitem__(name)

    def list(self):
        return tuple(n for n in self.fields.keys())

    def __iter__(self):
        return iter(self.fields)

    def __next__(self):
        return next(self.fields)
    """
    def search(self): #is search similar to get & get_name?
        pass
    """
    def __len__(self):
        return len(self.fields)

    def get_spans(self):
        """
        Return the name and spans of each field as a dictionary.
        """
        spans={}
        for name,field in self.fields.items():
            spans[name]=field.get_spans()
        return spans

    def apply_filter(self,filter_to_apply,ddf=None):
        """
        Apply the filter to all the fields in this dataframe, return a dataframe with filtered fields.

        :param filter_to_apply: the filter to be applied to the source field, an array of boolean
        :param ddf: optional- the destination data frame
        :returns: a dataframe contains all the fields filterd, self if ddf is not set
        """
        if ddf is not None:
            if not isinstance(ddf,DataFrame):
                raise TypeError("The destination object must be an instance of DataFrame.")
            for name, field in self.fields.items():
                # TODO integration w/ session, dataset
                newfld = field.create_like(ddf.name,field.name)
                ddf.add(field.apply_filter(filter_to_apply,dstfld=newfld),name=name)
            return ddf
        else:
            for field in self.fields.values():
                field.apply_filter(filter_to_apply)
            return self


    def apply_index(self, index_to_apply, ddf=None):
        """
        Apply the index to all the fields in this dataframe, return a dataframe with indexed fields.

        :param index_to_apply: the index to be applied to the fields, an ndarray of integers
        :param ddf: optional- the destination data frame
        :returns: a dataframe contains all the fields re-indexed, self if ddf is not set
        """
        if ddf is not None:
            if not isinstance(ddf, DataFrame):
                raise TypeError("The destination object must be an instance of DataFrame.")
            for name, field in self.fields.items():
                #TODO integration w/ session, dataset
                newfld = field.create_like(ddf.name, field.name)
                ddf.add(field.apply_index(index_to_apply,dstfld=newfld), name=name)
            return ddf
        else:
            for field in self.fields.values():
                field.apply_index(index_to_apply)
            return self

