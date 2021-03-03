# Import packages
import os
import cv2
import sys
import shutil
import argparse
import numpy as np
from imutils import paths, convenience

workpath = os.path.split(os.path.realpath(__file__))[0]

def blur_detection(
    images, 
    index,
    output=workpath + '/bd_result',
    save_img=False,
    resize=True):

    if save_img:
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

    result = dict()

    data = dict()
    for idx in index:
        image = images[idx]
        imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        if resize:
            imageGray = convenience.resize(imageGray, width=int(image.shape[1]//3.34))

        var = cv2.Laplacian(imageGray,cv2.CV_64F).var()

        if save_img:
            cv2.putText(image, str(var), (5,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.imwrite(output + '/' + str(idx) + '_' + str(var) + '.jpg', image)

        result[idx] = var

    result = sorted(result.items(), key=lambda item:item[0])
       
    return result

if __name__ == "__main__":
    pass