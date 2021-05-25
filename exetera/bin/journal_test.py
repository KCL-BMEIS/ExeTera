import time
from exetera.core.load_schema import load_schema
from exetera.core.session import Session
from exetera.core.journal import journal_test_harness

if __name__ == "__main__":
    schema_fn = '/home/ben/covid/covid_schema.json'
    old_fn = '/home/ben/covid/ds_20200801_base.hdf5'
    new_fn = '/home/ben/covid/ds_20200901_base.hdf5'
    dest_fn = '/home/ben/covid/ds_journal.hdf5'
    
    with open(schema_fn) as f:
        schema = load_schema(f)
    journal_test_harness(Session(), schema, old_fn, new_fn, dest_fn)
