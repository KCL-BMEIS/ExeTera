
from . import core, processing

from ._version import __version__


from .core.abstract_types import DataFrame
from .core.field_importers import ImporterDefinition
from .core.csv_reader_speedup import read_file_using_fast_csv_reader
from typing import Mapping, List
import csv
import numpy as np



def read_csv(filepath: str, 
             ddf: DataFrame,
             field_mapping: Mapping[str, ImporterDefinition],
             include: List[str] = None,
             exclude: List[str] = None,
             chunk_row_size: int = 1 << 20):
    """


    :params filepath:
    :params ddf:
    :params field_mapping:
    :params chunk_row_size:
    """

    # params validation
    if not isinstance(ddf, DataFrame):
        raise TypeError("The destination object must be an instance of DataFrame.")
        
    if not isinstance(field_mapping, dict):
        raise TypeError("'field_mapping' must be of type dict but is {}").format(type(field_mapping))
    
    field_mapping = {k.strip(): v for k, v in field_mapping.items()}

    # how to define timestamp??
    ts = None


    with open(filepath, encoding='utf-8') as sf:
        csvf = csv.DictReader(sf, delimiter=',', quotechar='"')
        csvf_fieldnames = [k.strip() for k in csvf.fieldnames]

        if field_mapping is not None: 
            fields = field_mapping.keys()
        else: 
            # schema_file {TODO}
            fields = None

        # validate all field name is defined in field_mapping
        missing_names = set(csvf_fieldnames).difference(set(fields))
        if len(missing_names) > 0:
            msg = "The following fields are present in file '{}' but not part of the 'field_mapping': {}"
            raise ValueError(msg.format(filepath, missing_names))    

        # check if included fields are in the file
        if include is not None and len(include) > 0:
            include_missing_names = set(include).difference(set(csvf_fieldnames))
            if len(include_missing_names) > 0:
                msg = "The following include fields are not part of the {}: {}"
                raise ValueError(msg.format(filepath, include_missing_names))

        # check if excluded fields are in the file
        if exclude is not None and len(exclude) > 0:
            exclude_missing_names = set(exclude).difference(set(csvf_fieldnames))
            if len(exclude_missing_names) > 0:
                msg = "The following exclude fields are not part of the {}: {}"
                raise ValueError(msg.format(filepath, exclude_missing_names))

        # get list of fields to be used
        fields_to_use = csvf_fieldnames
        if include is not None:
            fields_to_use = [k for k in fields_to_use if k in set(include)]
        if exclude is not None:
            fields_to_use = [k for k in fields_to_use if k not in set(exclude)]

        index_map = [csvf_fieldnames.index(k) for k in fields_to_use]

        field_importer_list = list() # only for field_to_use     
        for field_name in fields_to_use:
            importer_definition = field_mapping[field_name]
            field_importer = importer_definition.importer(ddf.dataset.session, ddf, field_name, ts)
            field_importer_list.append(field_importer)

        column_offsets = np.zeros(len(csvf_fieldnames) + 1, dtype=np.int64)
        for i, field_name in enumerate(csvf_fieldnames):
            importer_definition = field_mapping[field_name]
            column_offsets[i + 1] = column_offsets[i] + importer_definition.field_size * chunk_row_size
    
    read_file_using_fast_csv_reader(filepath, chunk_row_size, column_offsets, index_map, field_importer_list, stop_after_rows = None)

