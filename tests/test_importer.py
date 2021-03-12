import unittest
import json
import tempfile
import os
import h5py
from datetime import datetime, timezone
from exetera.core import importer


TEST_SCHEMA = json.dumps({
    'schema': {
        'schema_key': {
            "primary_keys": [
                'id'
            ],
            "fields": {
                'name': {
                    'field_type': 'string'
                },
                'id': {
                    'field_type': 'numeric',
                    'value_type': 'int32'
                },
                'timestamp':{
                    'field_type': 'datetime'
                }
            }
        }
    }
})


TEST_CSV_CONTENTS = '\n'.join((
    'name, id, timestamp',
    'a, 1, 2020-05-15:00-00-00',
    'b, 2, ',
    'c, 3, 2021-01-01:12-00-00'
))


class TestImporter(unittest.TestCase):
    def setUp(self):
        self.fd_schema, self.schema_file_name = tempfile.mkstemp(suffix='.json')
        with open(self.schema_file_name, 'w') as fschema:
            fschema.write(TEST_SCHEMA)

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(self.csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        self.files = {'schema_key': self.csv_file_name}


    def test_importer_with_arg_include(self):

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {'schema_key': ['id', 'name']}, {}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue(set(f['schema_key'].keys()) >= set(['id', 'name']))
            self.assertEqual(f['schema_key']['id']['values'].shape[0], 3)

        finally:
            os.close(fd_dest)

    def test_importer_with_wrong_arg_include(self):

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {'schema_wrong_key': ['id', 'name']}, {}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
        except Exception as e:
            self.assertEqual(str(e), "--include: the following include table(s) are not part of any input files: {'schema_wrong_key'}")
        finally:
            os.close(fd_dest)
            

    def test_importer_with_arg_exclude(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {}, {'schema_key':['timestamp']}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue(set(['timestamp']) not in set(f['schema_key'].keys()))
            self.assertEqual(f['schema_key']['id']['values'].shape[0], 3)

        finally:
            os.close(fd_dest)       


    def tearDown(self):
        os.close(self.fd_schema)
        os.close(self.fd_csv)