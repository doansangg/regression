import math
import numpy as np
import cv2
import random
from configs import *

def rotate_bound(image, angle, points=np.array([])):
    """
    rotate image in absolute points in image
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # print(angle, cos, sin)

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image

    new_image = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)


    if points.any():
        res = M.dot((np.hstack([points, np.ones(shape=(len(points), 1))])).T).T
        return new_image , res
    else:
        return new_image
        
def resize_image(image, width, height, borderMode=cv2.BORDER_REPLICATE):
    """Resize image keep image's aspect ratio by padded border

    Args:
        image (cv2 image): [description]
        width ([type]): [description]
        height ([type]): [description]
        borderMode ([type], optional): [description]. Defaults to cv2.BORDER_REPLICATE.
    """
    assert type(width) == int
    assert type(height) == int
    h, w = image.shape[:2]
    pad_bot, pad_right = 0, 0
    if w/h > width/height:
        new_w = width
        new_h = int(h*(new_w/w))
        pad_bot = height - new_h
    else:
        new_h = height
        new_w = int(w*(new_h/h))
        pad_right = width - new_w
    image = cv2.resize(image, (new_w, new_h))
    res = cv2.copyMakeBorder(image, 0, pad_bot, 0, pad_right, cv2.BORDER_REPLICATE)
    return res


def clockwise_points(points, refvec = [1, 0]):
    origin = np.average(points, axis=0)
    def clockwiseangle_and_distance(point):
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    return sorted(points, key=clockwiseangle_and_distance)

def process_img_lb(image,lb):
    lb_des = []
    h, w = image.shape[:2]
    # rotate iamge
    if random.random() < 0.5:
        image = cv2.blur(image, (5,5))
    ang = random.choice([-90, 0, 90, 180])
    # ang = 20
    lb[:, 0] = lb[:, 0]*w
    lb[:, 1] = lb[:, 1]*h
    image, lb = rotate_bound(image, ang, lb)
    lb = np.array(lb, dtype=np.float)
    # print(lb)

    nh, nw = image.shape[:2]
    lb = np.array(clockwise_points(lb))
    lb[:, 0] = lb[:, 0]/nw
    lb[:, 1] = lb[:, 1]/nh


#######
    # rescale image
    h, w = image.shape[:2]
    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
    nh, nw = image.shape[:2]
    lb[:, 0] = lb[:, 0]*(w/nw)*(nh/h) if w/h<nw/nh else lb[:, 0]
    lb[:, 1] = lb[:, 1]*(h/nh)*(nw/w) if w/h>nw/nh else lb[:, 1]
    #lb = lb.tolist()
        #lb_des = lb[0]+lb[1]+lb[2]+lb[3]
    #print(lb)
    lb = np.reshape(lb,(-1,))
    return image,lb