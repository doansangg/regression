import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from torch import Tensor, int32
import os
from helps import rotate_bound, resize_image, clockwise_points, process_img_lb
import cv2
from configs import *

class Dataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, path_data):
        self.path_img = path_data+"/"+ "image"
        self.path_label = path_data+"/"+ "label"
        self.images = []
        self.labels = []
        for item in os.listdir(self.path_img):
            self.images.append(self.path_img+"/"+item)
            self.labels.append(self.path_label+"/"+item.replace(".jpg",".txt"))
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            #transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        
        image = cv2.imread(self.images[index])
        label = self.process_label(self.labels[index])
        image, label = process_img_lb(image, label)
        image = self.rgb_loader(image)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
        return (image, torch.Tensor(label))
    
    def process_label(self,label_path):
        lb = []
        with open(label_path) as reader:
            coords = reader.read()
            coords = coords.split(',')[1:9]
            # print('coords: ', coords)
            for i in range(4):
                lb.append( [float(coords[i]), float(coords[i+4])])
            # print('pair: ', lb)
        lb = np.array(lb)
        return lb

    def rgb_loader(self, opencv_image):
        color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_converted)
        return pil_image
    
    def __len__(self):
        return self.size


def get_loader_train(path_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = Dataset(path_root)
    size = dataset.size
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, size


# # test dataloader
# datatrain = get_loader_train("./data_real",1)
# for i, (inputs, labels) in enumerate(datatrain):
#     print(inputs)
#     print(labels)
