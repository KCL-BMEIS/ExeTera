import unittest
from io import BytesIO, StringIO
import json
from datetime import datetime, timezone
import random
import string
import tempfile
import os
from exetera.core import importer
import h5py

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


CSV_CONTENTS = '\n'.join((
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
            fcsv.write(CSV_CONTENTS)

        self.files = {'schema_key': self.csv_file_name}


    def test_importer_with_arg_include(self):

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include_fields, exclude_fields = tuple(['id', 'name']), ()

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include_fields, exclude_fields)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue(set(f['schema_key'].keys()) >= set(['id', 'name']))
            self.assertEquals(f['schema_key']['id']['values'].shape[0], 3)

        finally:
            os.close(fd_dest)
            

    def test_importer_with_arg_exclude(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include_fields, exclude_fields = (), tuple(['timestamp'])

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include_fields, exclude_fields)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue(set(['timestamp']) not in set(f['schema_key'].keys()))
            self.assertEquals(f['schema_key']['id']['values'].shape[0], 3)

        finally:
            os.close(fd_dest)       


    def tearDown(self):
        os.close(self.fd_schema)
        os.close(self.fd_csv)
