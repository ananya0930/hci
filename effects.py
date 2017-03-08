import cv2
import numpy as np

class Effects(object):

    def render(self, image):

        # load calibration data
        #with np.load('webcam_calibration_ouput.npz') as X:
            #mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

        # set up criteria, object points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)



        objp = np.zeros((7*10,3), np.float32)
        objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        h = 0
        w =0
        images = glob.glob('images3/*.jpg')
        #print "hello"
        for fname in images:
            img = cv2.imread(fname)
            #print "hello2"
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            h,w = img.shape[:2]
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (10,7),None)
            #print ret
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
                #cv2.imshow('img',img)
                #cv2.waitKey(1000)


        #print objpoints
        #print imgpoints

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

        # find grid corners in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        if ret == True:

            # project 3D points to image plane
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            # draw cube
            self._draw_cube(image, imgpts)

    def _draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)

        # draw floor
        cv2.drawContours(img, [imgpts[:4]],-1,(200,150,10),-3)

        # draw pillars
        for i,j in zip(range(4),range(4,8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

        # draw roof
        cv2.drawContours(img, [imgpts[4:]],-1,(200,150,10),3)
