import h5py

old_source = 'old_vaccine_doses.hdf5'
new_source = 'new_vaccine_doses.hdf5'
schema_key = 'vacccine_doses'

# old_source = 'old_vacc_hesitancy.hdf5'
# new_source = 'new_vacc_hesitancy.hdf5'
# schema_key = 'vaccine_hesitancy'

# compare if they have same fields
with h5py.File(old_source, 'r') as hf1:
	old_fields_set = set(hf1[schema_key].keys())

with h5py.File(new_source, 'r') as hf2:
	new_fields_set = set(hf2[schema_key].keys())

assert(old_fields_set == new_fields_set)

sorted_field_list = sorted(list(old_fields_set))


# compare field by field
with h5py.File(old_source, 'r') as hf1:
	d1 = {}
	for field in sorted_field_list:
		for k in list(hf1[schema_key][field].keys()):
			key = field + '@' + k
			value = hf1[schema_key][field][k][:]
			d1[key] = value

with h5py.File(new_source, 'r') as hf2:
	d2 = {}
	for field in sorted_field_list:
		for k in list(hf2[schema_key][field].keys()):
			key = field + '@' + k
			value = hf2[schema_key][field][k][:]
			d2[key] = value


assert(sorted(d1.keys()) == sorted(d2.keys()))

keys = sorted(d1.keys())

for key in keys:
	v1 = d1[key]
	v2 = d2[key]
	try:
		assert(len(v1) == len(v2))
	except Exception as e:
		print(key)
		print('old', len(v1), v1)
		print('new', len(v2), v2)
		raise
	
	length = len(v1)

	if key.startswith('j_valid_from') or key.startswith('j_valid_to'):
		continue

	for i in range(length):
		if v1[i] != v2[i]:
			print(key)
			print(i, v1[i], v2[i])
			break