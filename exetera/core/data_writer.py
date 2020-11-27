from threading import Thread

class DataWriter:

    @staticmethod
    def clear_dataset(parent_group, name):
        t = Thread(target=DataWriter._clear_dataset,
                   args=(parent_group, name))
        t.start()
        t.join()

    @staticmethod
    def _clear_dataset(field, name):
        del field[name]

    @staticmethod
    def _create_group(parent_group, name, attrs):
        group = parent_group.create_group(name)
        for k, v in attrs:
            try:
                group.attrs[k] = v
            except Exception as e:
                print(f"Exception {e} caught while assigning attribute {k} value {v}")
                raise
        group.attrs['completed'] = False

    @staticmethod
    def create_group(parent_group, name, attrs):
        t = Thread(target=DataWriter._create_group,
                   args=(parent_group, name, attrs))
        t.start()
        t.join()

    @staticmethod
    def write(group, name, field, count, dtype=None):
        if name not in group.keys():
            DataWriter._write_first(group, name, field, count, dtype)
        else:
            DataWriter._write_additional(group, name, field, count)

    @staticmethod
    def _write_first(group, name, field, count, dtype=None):
        if dtype is not None:
            if count == len(field):
                ds = group.create_dataset(
                    name, (count,), maxshape=(None,), chunks=(1 << 20,), dtype=dtype)
                ds[:] = field
            else:
                ds = group.create_dataset(
                    name, (count,), maxshape=(None,), chunks=(1 << 20,), dtype=dtype)
                ds[:] = field[:count]
        else:
            if count == len(field):
                group.create_dataset(name, (count,), maxshape=(None,), chunks=(1 << 20,),
                                     data=field)
            else:
                group.create_dataset(name, (count,), maxshape=(None,), chunks=(1 << 20,),
                                     data=field[:count])

    @staticmethod
    def write_first(group, name, field, count, dtype=None):
        t = Thread(target=DataWriter._write_first,
                   args=(group, name, field, count, dtype))
        t.start()
        t.join()

    @staticmethod
    def _write_additional(group, name, field, count):
        gv = group[name]
        gv.resize((gv.size + count,))
        if count == len(field):
            gv[-count:] = field
        else:
            gv[-count:] = field[:count]

    @staticmethod
    def write_additional(group, name, field, count):
        t = Thread(target=DataWriter._write_additional,
                   args=(group, name, field, count))
        t.start()
        t.join()

    @staticmethod
    def _flush(group):
        group.attrs['completed'] = True

    @staticmethod
    def flush(group):
        t = Thread(target=DataWriter._flush, args=(group,))
        t.start()
        t.join()
