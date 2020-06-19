
import utils
import processing


# TODO: base filter for all hard filtered things, or should they be blitzed
# from the dataset completely?

# TODO: postprocessing activities
# * assessment sort by (patient_id, created_at)
# * aggregate from assessments to patients
#   * was first unwell
#   * first assessment
#   * last assessment
#   * assessment count
#   * assessment index start
#   * assessment index end

def postprocess(dataset, data_scheme, process_schema, timestamp):

    chunksize = 1 << 18

    # post process patients
    # TODO: need an transaction table
    processing.calculate_age_from_year_of_birth(
        dataset['patients'], dataset['patients']['year_of_birth'],
        utils.valid_range_fac_inc(0, 110), 2020, timestamp, name='age')
