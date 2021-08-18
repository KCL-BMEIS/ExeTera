from io import StringIO
from typing import Mapping, List, Union
import csv
import numpy as np
from datetime import datetime, timezone

from exetera.core.abstract_types import DataFrame
from exetera.core.field_importers import ImporterDefinition, TimestampImporter, IndexedStringImporter, INDEXED_STRING_FIELD_SIZE
from exetera.core.csv_reader_speedup import read_file_using_fast_csv_reader
from exetera.io import load_schema
from exetera.core import operations as ops


def read_csv(csv_file: str, 
             ddf: DataFrame,
             schema_dictionary: Mapping[str, ImporterDefinition] = None,
             schema_file: Union[str, StringIO] = None,
             schema_key: str = None,
             include: List[str] = None,
             exclude: List[str] = None,
             chunk_row_size: int = 1 << 20,
             timestamp= datetime.now(timezone.utc).timestamp()):
    """
    Read a comma-separated values (csv) file into HDF5DataFrame.

    :params csv_file: string path for csv file.
    :params ddf: destination dataframe.
    :params schema_dictionary: provide schema in dictionary format. Default is None.
    :params schema_file: provide schema in file. Default is None.
    :params include: a list of included field names. Default is None.
    :params exclude: a list of excluded field names. Default in None.
    :params chunk_row_size: read file chunk by chunk. Row sizes in each chunk is chunk_row_size. Default is 1 << 20.
    :params timestamp: timestamp. Default is timestamp of current time.
    """
    # params validation
    if not isinstance(ddf, DataFrame):
        raise TypeError("The destination object must be an instance of DataFrame.")
        
    if (schema_dictionary is None and schema_file is None) or (schema_dictionary is not None and schema_file is not None):
        raise ValueError("'schema_dict' and 'schema_file', one and only one of them should be provided.")

    if schema_dictionary is not None and not isinstance(schema_dictionary, dict):
        raise TypeError("'schema_dict' must be of type dict but is {}").format(type(schema_dictionary))

    if schema_file is not None and not isinstance(schema_file, (str, StringIO)):
        raise TypeError("'schema_file' must be of type str or StringIO but is {}").format(type(schema_file))

    # get schema_dict
    if schema_dictionary is not None:
        schema_dict = schema_dictionary
    else:
        # schmea_json_file is not None
        schemas = load_schema.load_schema(schema_file)
        if len(schemas) == 1:
            schema_dict = list(schemas.values())[0]
        elif len(schemas) > 1:
            if schema_key is None:
                raise ValueError("'schema_key' must be provided when there's multiple schemas in the schema file.")
            elif schema_key not in schemas:
                raise ValueError(f"'schema_key' must be provided correctly, but '{schema_key}' doesn't exist in the schema_file")
            else:
                schema_dict = schemas[schema_key] 
        else:
            raise ValueError("'schema_file' must not be empty.")

    read_csv_with_schema_dict(csv_file, ddf, schema_dict, timestamp, include, exclude, chunk_row_size)


def read_csv_with_schema_dict(csv_file: str, 
                              ddf: DataFrame,
                              schema_dictionary: Mapping[str, ImporterDefinition],
                              timestamp: float,
                              include: List[str] = None,
                              exclude: List[str] = None,
                              chunk_row_size: int = 1 << 20,
                              stop_after_rows = None
                              ):
    """
    Read a comma-separated values (csv) file into HDF5DataFrame, with schema provided in dictionary formats.

    :params csv_file: string path for csv file.
    :params ddf: destination dataframe.
    :params schema_dictionary: provide schema in dictionary format.
    :params timestamp: timestamp. 
    :params include: a list of included field names. Default is None.
    :params exclude: a list of excluded field names. Default in None.
    :params chunk_row_size: read file chunk by chunk. Row sizes in each chunk is chunk_row_size. Default is 1 << 20.
    :params stop_after_rows: stop after given rows. Default is None.
    """
    # get field_mapping
    field_mapping = {k.strip(): v for k, v in schema_dictionary.items()}

    with open(csv_file, encoding='utf-8') as sf:
        csvf = csv.DictReader(sf, delimiter=',', quotechar='"')
        csvf_fieldnames = [k.strip() for k in csvf.fieldnames]

        fields = field_mapping.keys()

        # validate all field name is defined in schema_dictionary
        undefined_names = set(csvf_fieldnames).difference(set(fields))
        if len(undefined_names) > 0:
            msg = "Warning: The following fields are present in file '{}' but not part of the 'schema_dictionary': {}"
            print(msg.format(csv_file, undefined_names))  

        # check if included fields are in the file
        if include is not None and len(include) > 0:
            include_missing_names = set(include).difference(set(csvf_fieldnames))
            if len(include_missing_names) > 0:
                msg = "The following include fields are not part of the {}: {}"
                raise ValueError(msg.format(csv_file, include_missing_names))

        # check if excluded fields are in the file
        if exclude is not None and len(exclude) > 0:
            exclude_missing_names = set(exclude).difference(set(csvf_fieldnames))
            if len(exclude_missing_names) > 0:
                msg = "The following exclude fields are not part of the {}: {}"
                raise ValueError(msg.format(csv_file, exclude_missing_names))

        # get list of fields to be used
        fields_to_use = csvf_fieldnames
        if include is not None:
            fields_to_use = [k for k in fields_to_use if k in set(include)]
        if exclude is not None:
            fields_to_use = [k for k in fields_to_use if k not in set(exclude)]

        index_map = [csvf_fieldnames.index(k) for k in fields_to_use]

        field_importer_list = list() # only for field_to_use     
        for field_name in fields_to_use:
            if field_name in undefined_names:
                field_importer = IndexedStringImporter(ddf.dataset.session, ddf, field_name, timestamp) 
            else:
                importer_definition = field_mapping[field_name]
                field_importer = importer_definition._importer(ddf.dataset.session, ddf, field_name, timestamp)
            field_importer_list.append(field_importer)

        column_offsets = np.zeros(len(csvf_fieldnames) + 1, dtype=np.int64)
        for i, field_name in enumerate(csvf_fieldnames):
            if field_name in undefined_names:
                field_size = INDEXED_STRING_FIELD_SIZE
            else:  
                importer_definition = field_mapping[field_name]
                field_size = importer_definition._field_size
            column_offsets[i + 1] = column_offsets[i] + field_size * chunk_row_size
    
    total_rows = read_file_using_fast_csv_reader(csv_file, chunk_row_size, column_offsets, index_map, field_importer_list, stop_after_rows)

    # create 'j_valid_from', 'j_valid_to' field in the end
    jvf_field_importer = TimestampImporter(ddf.dataset.session, ddf, 'j_valid_from', timestamp)
    valid_froms = np.full(total_rows, timestamp, dtype='float64')
    jvf_field_importer.write(valid_froms)

    jvt_field_importer = TimestampImporter(ddf.dataset.session, ddf, 'j_valid_to', timestamp)
    valid_tos = np.full(total_rows, ops.MAX_DATETIME.timestamp(), dtype='float64')
    jvt_field_importer.write(valid_tos)