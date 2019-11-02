import cv2 as cv
import numpy as np

class fix(object):
    def __init__(self):
        pass
    def load_params(self):
        self.mtx=np.load("mtx.npy")
        self.dist=np.load("dist.npy")
    def undistort(self,image):
        h,w=image.shape[:2]
        newcameramtx, roi= cv.getOptimcalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
        return cv2.undistort(image,self.mtx,self.dist,None,newcameramtx)

fixer=fix()
fixer.load_params()
