from exetera.core import fields as fld
from datetime import  datetime,timezone

class DataFrame():
    """
    DataFrame is a table of data that contains a list of Fields (columns)
    """
    def __init__(self, data=None):
        if data is not None:
            if isinstance(data,dict) and isinstance(list(a.items())[0][0],str) and isinstance(list(a.items())[0][1], fld.Field) :
                self.data=data
        self.data = dict()

    def add(self,field,name=None):
        if name is not None:
            if not isinstance(name,str):
                raise TypeError("The name must be a str object.")
            else:
                self.data[name]=field
        self.data[field.name]=field #note the name has '/' for hdf5 object

    def __contains__(self, name):
        """
        check if dataframe contains a field, by the field name
        name: the name of the field to check,return a bool
        """
        if not isinstance(name,str):
            raise TypeError("The name must be a str object.")
        else:
            return self.data.__contains__(name)

    def contains_field(self,field):
        """
        check if dataframe contains a field by the field object
        field: the filed object to check, return a tuple(bool,str). The str is the name stored in dataframe.
        """
        if not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object")
        else:
            for v in self.data.values():
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
            return self.data[name]

    def get_field(self,name):
        return self.__getitem__(name)

    def get_name(self,field):
        """
        Get the name of the field in dataframe
        """
        if not isinstance(field,fld.Field):
            raise TypeError("The field argument must be a Field object.")
        for name,v in self.data.items():
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
            self.data[name]=field
            return True

    def __delitem__(self, name):
        if not self.__contains__(name=name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self.data[name]
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
        return tuple(n for n in self.data.keys())

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        return next(self.data)
    """
    def search(self): #is search similar to get & get_name?
        pass
    """
    def __len__(self):
        return len(self.data)

    def apply_filter(self,filter_to_apply,dst):
        pass

    def apply_index(self, index_to_apply, dest):
        pass

    def sort_on(self,dest_group, keys,
                timestamp=datetime.now(timezone.utc), write_mode='write', verbose=True):
        pass