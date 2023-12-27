import sys
sys.path.append('/teamspace/studios/this_studio/AgingGAN/')
from Data.Dataset import IMDBDataset
import numpy as np
import logging

def find_mean_std_of_dataset(n=20000):
    dataset = IMDBDataset('/teamspace/studios/this_studio/imdb_crop', '/teamspace/studios/this_studio/imdb/imdb.mat', None)
    mean = [0] * 3
    std = [0] * 3
    print('Calculating mean and std...')

    for i, (image, _) in enumerate(dataset):
        print(str(i) + '/' + str(n))
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]
        mean[0] += np.mean(r_channel)
        mean[1] += np.mean(g_channel)
        mean[2] += np.mean(b_channel)

        std[0] += np.std(r_channel)
        std[1] += np.std(g_channel)
        std[2] += np.std(b_channel)
        if n == i:
            break
    
    mean = [i / n for i in mean]
    std = [i / n for i in std]

    return mean, std

print(find_mean_std_of_dataset())