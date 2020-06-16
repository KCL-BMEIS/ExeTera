
import utils
import processing





def postprocess(dataset, data_scheme, process_schema, timestamp):

    chunksize = 1 << 18

    # post process patients
    # TODO: need an transaction table
    processing.calculate_age_from_year_of_birth(
        dataset['patients'], dataset['patients']['year_of_birth'],
        utils.valid_range_fac_inc(0, 110), 2020, timestamp, name='age')
