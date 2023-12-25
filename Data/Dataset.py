from torch.utils.data import Dataset
import os
import scipy
import sys
sys.path.append('/teamspace/studios/this_studio')
sys.path.append('/teamspace/studios/this_studio/utilities')
from utilities.utils import get_medatada_information

class IMDBDataset(Dataset):
    def __init__(self, img_directory, metadata_file, transforms):
        self.img_directory = img_directory
        self.metadata_file = metadata_file
        self.transforms = transforms
        self.images = self.open_directory()

    def __len__(self):
        return len(self.images)

    def open_directory(self):
        images = list()
        for dir in os.listdir(self.img_directory):
            for img in os.listdir(self.img_directory + '/' + dir):
                images.append(self.img_directory + '/' + dir + '/' + img)
        
        return sorted(images)

    def __getitem__(self, idx):
        image = self.images[idx]
        age = get_medatada_information(self.metadata_file, image)
        if self.transforms:
            image = self.transforms(image)

        return image, age

dataset = IMDBDataset('/teamspace/studios/this_studio/imdb_crop', '/teamspace/studios/this_studio/imdb/imdb.mat', None)
