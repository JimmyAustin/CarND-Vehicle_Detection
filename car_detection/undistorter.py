import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

class Undistorter(object):
    def __init__(self, calibration_images):
        self.mtx, self.dist = Undistorter.get_undistort_matrix(calibration_images)

    def get_undistort_matrix(calibration_images, ny=6, nx=9):
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2).astype('float32')

        objpoints = []
        imgpoints = []

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        img_size = (calibration_images[0].shape[1], calibration_images[1].shape[0])

        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

        return mtx, dist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def load_from_path(path):
        calibration_imgs = [cv2.imread(join(path, f)) for f in listdir(path) if isfile(join(path, f))]
        return Undistorter(calibration_imgs)
