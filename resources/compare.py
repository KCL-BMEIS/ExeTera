import exetera
from exetera.core.session import Session
import numpy as np

# old_source = 'resources/old_vaccine_doses.hdf5'
# new_source = 'resources/new_vaccine_doses.hdf5'
# schema_key = 'vaccine_doses'

# old_source = 'resources/old_vaccine_symptoms.hdf5'
# new_source = 'resources/new_vaccine_symptoms.hdf5'
# schema_key = 'vaccine_symptoms'

# old_source = 'resources/old_vaccine_hesitancy.hdf5'
# new_source = 'resources/new_vaccine_hesitancy.hdf5'
# schema_key = 'vaccine_hesitancy'

# old_source = 'resources/old_mental_health.hdf5'
# new_source = 'resources/new_mental_health.hdf5'
# schema_key = 'mental_health'

# old_source = 'resources/old_patients.hdf5'
# new_source = 'resources/new_patients.hdf5'
# schema_key = 'patients'

# old_source = 'resources/old_diet.hdf5'
# new_source = 'resources/new_diet.hdf5'
# schema_key = 'diet'

# old_source = 'resources/old_tests.hdf5'
# new_source = 'resources/new_tests.hdf5'
# schema_key = 'tests'

old_source = 'resources/old_assessments.hdf5'
new_source = 'resources/new_assessments.hdf5'
schema_key = 'assessments'


with Session() as s:
    old = s.open_dataset(old_source, 'r', 'ds1')
    new = s.open_dataset(new_source, 'r', 'ds2')
    olddf = old[schema_key]
    newdf = new[schema_key]

    keys = olddf.keys()
    # keys = ['diabetes_oral_other_medication','outward_postcode_region', 'vs_other']
    # keys = ['supplements_other']


    for k in keys:
        old_ = olddf[k].data[:]
        new_ = newdf[k].data[:]
        if k.startswith('j_valid_from') or k.startswith('j_valid_to'):
            continue

        # if not k.startswith('outward_postcode'):
        #     continue

        for i in range(len(old_)):   
        
            if old_[i] != new_[i]:
                print('======')
                print(k, i)
                print('old: ', old_[i])
                print('*******')
                print('new: ', new_[i])

                for j in range(len(old_[i])):
                    if old_[i][j] != new_[i][j]:
                        print(j, old_[i][j], new_[i][j])


        # if olddf[k].indexed:
        #     # print(k, len(olddf[k].indices[:]), len(newdf[k].indices[:])) 
        #     indice_equal = np.array_equal(olddf[k].indices[:], newdf[k].indices[:])
        #     values_equal = np.array_equal(olddf[k].values[:], newdf[k].values[:])
        #     # data_equal = np.array_equal(olddf[k].data[:], newdf[k].data[:])

        #     if indice_equal == False or values_equal == False:
        #         print(olddf[k], k, indice_equal, values_equal)
        #         print('old_length', len(olddf[k]), 'new_length', len(newdf[k]))

                # old_ins = olddf[k].indices[:]
                # new_ins = newdf[k].indices[:]

                # for i in range(len(olddf[k])):   
                    
                #     if old_ins[i] != new_ins[i]:
                #         print('======')
                #         print(k, i)
                #         print('old: ', old_ins[i])
                #         print('*******')
                #         print('new: ', new_ins[i])
                #         break


        # else:
        #     data_equal = np.array_equal(olddf[k].data[:], newdf[k].data[:])
        #     if not data_equal:
        #         print(olddf[k], k)



                # for x in range(110):
                #     print(x, old_[i][x], new_[i][x])


        # if k == 'vs_other':
        # line_no = 1564504
        # old_start = olddf[k].indices[line_no]
        # old_end = olddf[k].indices[line_no + 1]

        # new_start = newdf[k].indices[line_no]
        # new_end = newdf[k].indices[line_no + 1]

        # x = olddf[k].values[old_start: old_end]
        # y = newdf[k].values[new_start: new_end]

        # print(x)
        # print(y)

        # for i in range(len(x)):
        #     if x[i] != y[i]:
        #         print(i, x[i], y[i])
        


                