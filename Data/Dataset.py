from torch.utils.data import Dataset
import os
import cv2
import sys
import scipy
sys.path.append('/teamspace/studios/this_studio')
sys.path.append('/teamspace/studios/this_studio/utilities')
from utilities.utils import get_medatada_information

class IMDBDataset(Dataset):
    def __init__(self, img_directory, metadata_file, transforms):
        self.img_directory = img_directory
        self.metadata_file = metadata_file
        self.transforms = transforms
        self.images = self.open_directory()
        self.data = scipy.io.loadmat(metadata_file)

    def __len__(self):
        return len(self.images)

    def open_directory(self):
        images = list()
        for dir in os.listdir(self.img_directory):
            for img in os.listdir(self.img_directory + '/' + dir):
                images.append(self.img_directory + '/' + dir + '/' + img)
        
        return sorted(images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        age = get_medatada_information(self.data, image_path)
        if self.transforms:
            image = self.transforms(image)

        return image, age