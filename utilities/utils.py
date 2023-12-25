import datetime as dt
import numpy as np
import scipy

NUMBER_OF_DAYS = 365

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return dt.date.fromordinal(int(datenum)) \
           + dt.timedelta(days=days) \
           - dt.timedelta(days=366)
        
def get_medatada_information(metadata_file, image_path):
    data = scipy.io.loadmat(metadata_file)

    path_list = image_path.split('/')[-2:]
    path = '/'.join(path_list)
    path_array = np.array(path)

    image_paths = data['imdb'][0][0][2][0]

    pos = np.where(image_paths == path_array)
    if len(pos) < 1:
        raise ValueError("Path was not found in metadata!")
    elif len(pos) > 1:
        raise ValueError("Multiple paths were found in metadata!")

    date_of_birth = datenum_to_datetime(int(data['imdb'][0][0][0][0][pos[0][0]]))
    date_of_picture = dt.date(data['imdb'][0][0][1][0][pos[0][0]], 7, 1)
    actual_age = date_of_picture - date_of_birth
    
    return actual_age.days // NUMBER_OF_DAYS

