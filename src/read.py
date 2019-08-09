import cv2
import numpy as np
import imutils
from imutils import contours

# load image, calculate the ratio and resize to 300px
file_path = '../data/test4.jpg'

img = cv2.imread(file_path)
ratio = img.shape[0] / 300.0
orig = img.copy()
img = imutils.resize(img, height = 300)

# transform the image
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
denoised = cv2.fastNlMeansDenoising(gray)
edges = cv2.Canny(denoised,90,150,apertureSize = 3)

# find the sudoku board in the image
cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
board_cnt = None
	
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015 * peri, True)

	if len(approx) == 4:
		board_cnt = approx
		break

# transform the board to birds-eye view
pts = board_cnt.reshape(4,2)
rect = np.zeros((4,2), dtype="float32")

# compute placement of the edges bassed on cooridinates (x,y)
# where the (0,0) point is in the top left of the image
# top-left has minimum sum, top-right has minimum difference
# bottom-left has maximum difference and bottom-roght has maximum sum
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

# multiply by ratio in order to retrieve the board in original size
rect *= ratio

# calculate the size of the board using euqlidean distance
tl, tr, bl, br = rect

width_top = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
width_bottom = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
height_left = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
heigth_right = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)

max_width = max(int(width_top), int(width_bottom))
max_heigth = max(int(height_left), int(heigth_right)) 

mapping = np.array([
    [0,0], 
    [max_width - 1, 0], 
    [max_width - 1, max_heigth - 1], 
    [0, max_heigth - 1]], 
    dtype = "float32")

# perform the transformation
mapped_img = cv2.getPerspectiveTransform(rect, mapping)
warped_img = cv2.warpPerspective(orig, mapped_img, (max_width, max_heigth))

# process the warped image
warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
# warped_denoised = cv2.fastNlMeansDenoising(warped_gray)
# warped_edges = cv2.Canny(warped_denoised,90,150,apertureSize = 3)
warped_gray = cv2.bitwise_not(warped_gray)
warped_bw = cv2.adaptiveThreshold(warped_gray, 
                                  255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  15,
                                  -2)

horizontal = warped_bw.copy()
vertical = warped_bw.copy()

# prepare the structure to remove horizontal lines
cols = horizontal.shape[1]
horizontal_size = cols // 10
horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontal_structure)
horizontal = cv2.dilate(horizontal, horizontal_structure)

# prepare the sturcture to remove vertical lines
rows = vertical.shape[0]
vertical_size = rows // 10
vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, vertical_structure)
vertical = cv2.dilate(vertical, vertical_structure)

testim = warped_bw - vertical
testim = testim - horizontal

smooth_test = testim.copy()
smooth_test = cv2.fastNlMeansDenoising(smooth_test, None, 40, 40)
smooth_test = cv2.blur(smooth_test, (2,2))

# cut image into 81 separate windows
height, width = smooth_test.shape

windowsize_r = height // 9
windowsize_c = width // 9

windows = []
for r in range(0,smooth_test.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,smooth_test.shape[1] - windowsize_c, windowsize_c):
        window = smooth_test[r:r+windowsize_r,c:c+windowsize_c]
        windows.append(window)

# test save
for i in range(len(windows)):
    cv2.imwrite('../windows/window_{}.png'.format(i), windows[i])