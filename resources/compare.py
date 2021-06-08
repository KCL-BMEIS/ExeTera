import exetera
from exetera.core.session import Session
import numpy as np

# old_source = 'resources/old_vaccine_doses.hdf5'
# new_source = 'resources/new_vaccine_doses.hdf5'
# schema_key = 'vacccine_doses'

# old_source = 'resources/old_vacc_hesitancy.hdf5'
# new_source = 'resources/new_vacc_hesitancy.hdf5'
# schema_key = 'vaccine_hesitancy'

# old_source = 'resources/old_mental_health.hdf5'
# new_source = 'resources/new_mental_health.hdf5'
# schema_key = 'mental_health'

old_source = 'resources/old_patients.hdf5'
new_source = 'resources/new_patients.hdf5'
schema_key = 'patients'


with Session() as s:
    old = s.open_dataset(old_source, 'r', 'ds1')
    new = s.open_dataset(new_source, 'r', 'ds2')
    olddf = old[schema_key]
    newdf = new[schema_key]

    keys = olddf.keys()
    keys = ['cancer_clinical_trial_site', 'cancer_type', 'clinical_study_institutions', 
            'clinical_study_names', 'clinical_study_nct_ids', 'se_postcode' ]
    # keys = ['diabetes_oral_other_medication','outward_postcode_region', 'vs_other']
    

    for k in keys:
        if k.startswith('j_valid_from') or k.startswith('j_valid_to'):
            continue

        if olddf[k].indexed:
            # print(k, len(olddf[k].indices[:]), len(newdf[k].indices[:])) 
            indice_equal = np.array_equal(olddf[k].indices[:], newdf[k].indices[:])
            values_equal = np.array_equal(olddf[k].values[:], newdf[k].values[:])
            # data_equal = np.array_equal(olddf[k].data[:], newdf[k].data[:])

            if indice_equal == False or values_equal == False:
                print(olddf[k], k, indice_equal, values_equal)

                print('old_length', len(olddf[k]), 'new_length', len(newdf[k]))

                old_ins = olddf[k].indices[:]
                new_ins = newdf[k].indices[:]

                for i in range(len(olddf[k])):   
                    
                    if old_ins[i] != new_ins[i]:
                        print('======')
                        print(k, i)
                        print('old: ', old_ins[i])
                        print('*******')
                        print('new: ', new_ins[i])
                        break


        # else:
        #     data_equal = np.array_equal(olddf[k].data[:], newdf[k].data[:])
        #     if not data_equal:
        #         print(olddf[k], k)



                # for x in range(110):
                #     print(x, old_[i][x], new_[i][x])


        # if k == 'bmi':
        #     line_no = 3595921
        #     old_start = olddf[k].indices[line_no]
        #     old_end = olddf[k].indices[line_no + 1]

        #     new_start = newdf[k].indices[line_no]
        #     new_end = newdf[k].indices[line_no + 1]

        #     print(olddf[k].values[old_start: old_end])
        #     print(newdf[k].values[new_start: new_end])
        


                