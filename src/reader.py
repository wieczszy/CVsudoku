import cv2
import imutils
import numpy as np
from imutils import contours


class ImageReader():
    def __init__(self):
        pass

    def _load_image(self, file_path):
        return cv2.imread(file_path)

    def _preprocess_input(self, image):
        image = imutils.resize(image, height=300)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        edges = cv2.Canny(denoised, 90, 150, apertureSize=3)
        return edges

    def _find_board_contour(self, image):
        cnts = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        board_cnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                board_cnt = approx
                break

        return board_cnt

    def _get_image_ratio(self, image):
        return image.shape[0] / 300.0

    def _calculate_board_coords(self, board_cnt, ratio):
        pts = board_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        rect *= ratio

        return rect

    def _extract_board(self, coords, input_image):
        tl, tr, bl, br = coords

        width_top = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
        width_bottom = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
        height_left = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        heigth_right = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)

        max_width = max(int(width_top), int(width_bottom))
        max_heigth = max(int(height_left), int(heigth_right))

        mapping = np.array([
            [0,0],
            [max_width - 1, 0],
            [max_width - 1, max_heigth - 1],
            [0, max_heigth - 1]],
            dtype="float32")

        mapped_img = cv2.getPerspectiveTransform(coords, mapping)
        warped_img = cv2.warpPerspective(input_image, mapped_img, (max_width, max_heigth))
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        warped_gray = cv2.bitwise_not(warped_gray)
        warped_bw = cv2.adaptiveThreshold(warped_gray,
                                        255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY,
                                        15,
                                        -2)

        return warped_bw

    def _remove_grid(self, image):
        horizontal = image.copy()
        vertical = image.copy()

        cols = horizontal.shape[1]
        horizontal_size = cols // 10
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.erode(horizontal, horizontal_structure)
        horizontal = cv2.dilate(horizontal, horizontal_structure)

        rows = vertical.shape[0]
        vertical_size = rows // 10
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.erode(vertical, vertical_structure)
        vertical = cv2.dilate(vertical, vertical_structure)

        cleaned = image - horizontal
        cleaned = cleaned - vertical

        cleaned = cv2.fastNlMeansDenoising(cleaned, None, 40, 40)
        cleaned = cv2.blur(cleaned, (2, 2))

        return cleaned

    def _split_to_cells(self, image):
        height, width = image.shape

        windowsize_rows = height // 9
        windowsize_cols = width // 9

        cells = []
        for r in range(0, image.shape[0] - windowsize_rows, windowsize_rows):
            for c in range(0, image.shape[1] - windowsize_cols, windowsize_cols):
                cell = image[r:r+windowsize_rows, c:c+windowsize_cols]
                cell = cv2.resize(cell, (28,28), 1)
                cells.append(cell)

        return cells

    def extract_board_cells(self, image_path):
        input_image = self._load_image(image_path)
        preprocessed_image = self._preprocess_input(input_image)
        board_contour = self._find_board_contour(preprocessed_image)
        original_ratio = self._get_image_ratio(input_image)
        board_coords = self._calculate_board_coords(board_contour, original_ratio)
        board_image = self._extract_board(board_coords, input_image)
        board_clened = self._remove_grid(board_image)
        cells = self._split_to_cells(board_clened)
        cells = np.array(cells)
        return cells
