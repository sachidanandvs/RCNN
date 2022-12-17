import cv2
import random
import numpy as np
import torch

def get_position(truncation,o_w,o_h):
    tmax = truncation[1]/100
    tmin = truncation[0]/100
    w, h = 28, 28
    val = random.random()
    if(o_w > 10):
        if(val < 0.25):
            x = random.randint(
                int(-(tmax) * o_w),
                int(-(tmin) * o_w),
            )
            y = random.randint(0, h - o_h)
        elif(0.25 <= val < 0.5):
            x = random.randint(
                w - o_w + int((tmin) * o_w),
                w - o_w + int((tmax) * o_w),
            )
            y = random.randint(0, h - o_h)
        elif(0.5 <= val < 0.75):
            y = random.randint(
                int(-(tmax) * o_h),
                int(-(tmin) * o_h),
            )
            x = random.randint(0, w - o_w)
        else:
            y = random.randint(
                h - o_h + int((tmin) * o_h),
                h - o_h + int((tmax) * o_h),
            )
            x = random.randint(0, w - o_w)
    else:
        if(val < 0.5):
            y = random.randint(
                int(-(tmax) * o_h),
                int(-(tmin) * o_h),
            )
            x = random.randint(0, w - o_w)
        else:
            y = random.randint(
                h - o_h + int((tmin) * o_h),
                h - o_h + int((tmax) * o_h),
            )
            x = random.randint(0, w - o_w)
    return x,y

def shift_image(image,tr):
    image = image.cpu().numpy().squeeze(0)
    W, H = image.shape
    # get bounding box around white pixels in image
    bimg = image.copy()
    bimg = ((bimg*np.array([0.1307]) + np.array([0.3081]))*255).astype(np.uint8)
    bimg = cv2.adaptiveThreshold(bimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)

    # cont = cv2.findNonZero(bimg)
    # x, y, w, h = cv2.boundingRect(cont)
    # cv2.rectangle(bimg, (x, y), (x+w, y+h), (255, 255, 255),2)
    
    cont = cv2.findNonZero(bimg)
    x1, y1, w1, h1 = cv2.boundingRect(cont)

    if(tr == [0,1]):
        x_l, y_l = x1, y1
    else:
        x_l, y_l = get_position(tr,w1,h1)
    # shift the image to edge in corresponding direction
    background = np.zeros((3*W, 3*H), np.uint8)
    background[W + x_l:W+x_l+w1, H+y_l:H+y_l+h1] = image[x1:x1+w1,y1:y1+h1]
    # cv2.rectangle(background,(W,H),(2*W,2*H),(200,200,200),1)
    # cv2.rectangle(background,(W+x_l,H+y_l),(W+x_l+w1,H+y_l+h1),(200,200,200),1)
    # plt.imshow(background,cmap="gray")
    # plt.show()
    return background[W:2*W,H:2*H]