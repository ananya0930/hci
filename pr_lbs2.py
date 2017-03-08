import cv2
from glyphdatabase import *
from glyphfunctions import *
from webcam import Webcam
import glob

#webcam = Webcam()
#webcam.start()

QUADRILATERAL_POINTS = 4
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
camera = cv2.VideoCapture(0)


# objp = np.zeros((7*10,3), np.float32)
# objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# h = 0
# w =0
#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# images = glob.glob('images3/*.jpg')
# #print "hello"
# for fname in images:
#     img = cv2.imread(fname)
#     #print "hello2"
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     h,w = img.shape[:2]
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (10,7),None)
#     #print ret
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         #img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
#         #cv2.imshow('img',img)
#         #cv2.waitKey(1000)
#
#
# #print objpoints
# #print imgpoints
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#
#
#
# matrix = mtx

# matrix = np.array([[632.09914295, 0.0, 326.3590625 ],
# 		   [0.0, 636.95321625, 270.72390718],
#  		   [0.0, 0.0, 1.0]])

matrix = np.array([[ 967.14084813, 0.0, 547.48661951],
            [ 0.0, 968.38540386, 399.77711243],
            [ 0.0, 0.0, 1.0 ]])


#dist = np.array([ 0.27540634, -1.2389006, 0.00627349, -0.00170199, 0.79555896]).reshape(5, 1)

dist = np.array([ 0.02505536, -0.1382031, 0.01193481, 0.00349419, 0.24150026]).reshape(5, 1)


objpoints = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64).reshape(4, 3, 1)
projpoints = np.array([[0,0,0],[0,0,-1],[0,1,0],[0,1,-1],[1,0,0],[1,0,-1],[1,1,0],[1,1,-1]], dtype=np.float64).reshape(8, 3, 1)


while True:


    ret, frame = camera.read()
    image = frame
    # Stage 1: Read an image from our webcam
    #image = webcam.get_current_frame()

    # Stage 2: Detect edges in image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 100, 200)

    # Stage 3: Find contours
    masks , contours, heirarchy= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:

        # Stage 4: Shape check
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04*perimeter, True)

        if len(approx) == QUADRILATERAL_POINTS:

            # Stage 5: Perspective warping
            topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))

            # Stage 6: Border check
            resized_shape = resize_image(topdown_quad, SHAPE_RESIZE)
            if resized_shape[5, 5] > BLACK_THRESHOLD: continue

            # Stage 7: Glyph pattern
            glyph_pattern = get_glyph_pattern(resized_shape, BLACK_THRESHOLD, WHITE_THRESHOLD)
            glyph_found, glyph_rotation, glyph_substitute = match_glyph_pattern(glyph_pattern)

            if glyph_found:

                # Stage 8: Substitute glyph
                substitute_image = cv2.imread('glyphs/images/faces.jpg'.format(glyph_substitute))

                for _ in range(glyph_rotation):
                    substitute_image = rotate_image(substitute_image, 90)

                image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))

                img = image

                imgpoints = np.array(approx, dtype=np.float64).reshape(4, 2, 1)
                ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, matrix, dist)
                imgpoints, jacobian = cv2.projectPoints(projpoints, rvec, tvec, matrix, dist)


                point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
                point2 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
                point2 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
                point2 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
                point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
                point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
                point2 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
                point2 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
                point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
                point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
                point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
                point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)

                point1 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
                point2 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)


    # Stage 9: Show augmented reality
    cv2.imshow('2D Augmented Reality using Glyphs', image)
    cv2.waitKey(10)
