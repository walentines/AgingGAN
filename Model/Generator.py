from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # declaring the components

        # embeddings of noise and condition
        self.cond_embedder = nn.Linear(1, 50)
        self.noise_embedder = nn.Linear(100, 200)
        self.global_embedder = nn.Linear(250, 500)

        # 1 x 1
        self.conv1 = nn.ConvTranspose2d(500, 64 * 16, 4)
        self.batch_1 = nn.BatchNorm2d(64 * 16)

        # 4 x 4
        self.conv2 = nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1)
        self.batch_2 = nn.BatchNorm2d(64 * 8)

        # 8 x 8
        self.conv3 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1)
        self.batch_3 = nn.BatchNorm2d(64 * 4)
        # 16 x 16
        self.conv4 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1)
        self.batch_4 = nn.BatchNorm2d(64 * 2)
        # 32 x 32
        self.conv5 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1)
        self.batch_5 = nn.BatchNorm2d(64)
        # 64 x 64
        self.conv6 = nn.ConvTranspose2d(64, 16, 4, 2, 1)
        self.batch_6 = nn.BatchNorm2d(16)
        # 128 x 128
        self.conv7 = nn.ConvTranspose2d(16, 3, 4, 2, 1)
        # 3 x 256 x 256
    
    def forward(self, x, y):
        # using the components
        # generation of noise should be done outside this forward 
        cond_embedding = nn.ReLU(self.cond_embedder(y)).unsqueeze(1)
        noise_embedding = nn.ReLU(self.noise_embedder(x)).unsqueeze(1)
        concatenation = torch.concat([cond_embedding, noise_embedding], axis = 1)
        global_embedding = nn.ReLU(self.global_embedder(concatenation)).unsqueeze(2)
        # 500 x 1 x 1
        
        return nn.Sigmoid(self.conv7(
                    nn.ReLU(self.batch_6(self.conv6(
                                nn.ReLU(self.batch_5(self.conv5(
                                            nn.ReLU(self.batch_4(self.conv4(
                                                            nn.ReLU(self.batch_3(self.conv3(
                                                                        nn.ReLU(self.batch_2(self.conv2(
                                                                                        nn.ReLU(self.batch_1(self.conv1
                                                                                                    (global_embedding))))))))))))))))))))
