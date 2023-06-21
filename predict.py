# Model class must be defined somewhere
import torch
from torchvision import transforms
from configs import *
import cv2
from PIL import Image

transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
            ])
image = cv2.imread("./data_real/image/29C67895-1678850783658.jpg")

def process_img(opencv_image):
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)
    img_t = transform(pil_image)
    return torch.unsqueeze(img_t, 0)


batch_t = process_img(image).to("cuda")
model = torch.load("best.pt").to("cuda")
model.train()
out = model(batch_t)
list_coor = out.tolist()[0]
print(list_coor)


# draw
colors = [(255, 0, 0), (0, 255, 0), (0, 0 , 255), (255, 255, 0)]
h, w = image.shape[:2]
for i in range(4):
    i=i*2
    cv2.circle(image, (int(list_coor[i]*w), int(list_coor[i+1]*h)), 3, colors[1], 3)
cv2.imshow("image",image)
cv2.waitKey(0)
