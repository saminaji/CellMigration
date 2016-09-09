import sys
import cv2
import numpy as np
import glob

# this script is aimed at matching contours in images
def extrac_ref_cont(img):
    ref = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(ref,127,255,0)
    cont = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]

    for contour in cont:
        area = cv2.contourArea(contour)
        imgArea = img.shape[0]*img.shape[1]

        if 0.05 < area/float(imgArea) <0.8:
            return contour

    return ref

def extract_all_cont(img):
    ref = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(ref,127,255,0) # get pixels whose max intensity is 127 in  the RGB color system
    conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2] # find contours by setting up the method to simple

    return conts

if __name__=='__main__':
    # read frames of the video
    path = glob.glob('/home/sami/Desktop/movies/movies/*.avi');
    cap = cv2.VideoCapture(path[0])
    timestamp = []
    count = 0
    frames = [];
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame == None:
                break;
            cv2.imshow('img',frame)
            time = cap.get(0)
            timestamp.append(time)
            frames.append(frame)

    except EOFError:
        pass

    for i in range(len(frames)):
        img1 = frames[0]  # reference image
        img2 = frames[i+1]  # input image
         # extract the reference contour to be found in the image
        ref_cont = extrac_ref_cont(img1)
         # exrtract all the contours in the image
        input_cont = extract_all_cont(img2)
         # closet contour
        closet_cont = input_cont[0]
        min_dist = sys.maxint

         # now lets find the closet contour
        for conts in input_cont:
             # matching the shape and take the closest one
            ret = cv2.matchShapes(ref_cont, conts, 1, 0.0)
            if ret < min_dist:
                min_dist = ret
                closet_cont = conts
        cv2.drawContours(img1, [closet_cont], -1, (0, 0, 0), 3)
        cv2.imshow('output', img2)

