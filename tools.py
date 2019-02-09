import sys, os
# sys.path.append(os.path.dirname("./PeakFinder/"))
sys.path.append(os.path.dirname("/cfs/klemming/nobackup/j/jcalven/lar/software/PeakFinder/"))
#from pax.formats import flat_data_formats
import h5py

def read_hdf5(file, fields_to_exclude=(), fields_to_include=(), show=False):
    """Read hdf5 data"""

    if not isinstance(fields_to_exclude, (list, tuple)):
        fields_to_exclude = [fields_to_exclude]

    if not isinstance(fields_to_include, (list, tuple)):
        fields_to_include = [fields_to_include]

    # reader = flat_data_formats['hdf5']()
    # reader.open(file, 'r')
    reader = h5py.File(file, 'r')

    data_types = list(reader.keys())  # data_types_present
    if show:
        print(data_types)
        return

    # Fields in 'fields_to_include' takes precedence over any field in
    # 'fields_to_exclude'
    if fields_to_include:
        out_dict = {key: None for key in data_types if key in fields_to_include}
    else:
        out_dict = {key: None for key in data_types if key not in fields_to_exclude}

    for key in out_dict.keys():
        out_dict.update({key: reader.get(key)[0::]})  # read_data(key)})
    reader.close()
    return out_dict
