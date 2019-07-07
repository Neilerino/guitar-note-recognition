import cv2 as cv
from math import inf
from statistics import median
import numpy as np

class Neck:

    def __init__(self, img):
        self.image = img
        grey = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.image = rotate_image(self.image)
        self.neck = crop_neck(self.image)

    def get(self):
        return self.neck

def rotate_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Sobel(gray, cv.CV_8U, 0, 1, 3)
    edges = threshold(edges, 127)

    threshold_value = 15
    min_length = 50
    min_gap = 50

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 15, min_length, min_gap)
    slopes = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append(abs((y2 - y1) / (x2 - x1)))

    median_slope = median(slopes)
    angle = median_slope * 45

    return rotate(img, angle)

def crop_neck(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Sobel(gray, cv.CV_8U, 0, 1, 3)
    edges = threshold(edges, 127)

    threshold_value = 15
    min_length = 50
    min_gap = 50

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 15, min_length, min_gap)

    y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            y.append(y1)
            y.append(y2)

    y_sort = list(sorted(y))
    y_differences = [0]

    first_y = 0
    last_y = inf

    for i in range(len(y_sort) - 1):
        y_differences.append(y_sort[i + 1] - y_sort[i])
    for i in range(len(y_differences) - 1):
        if y_differences[i] == 0:
            last_y = y_sort[i]
            if i > 3 and first_y == 0:
                first_y = y_sort[i]

    return img[first_y - 10:last_y + 10]

def threshold(img, s):
    I = img
    I[I <= s] = 0
    I[I > s] = 255
    return I


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated

def detect_frets(img):
    min_length = 4
    max_gap = 0
    blur = cv.GaussianBlur(img, (5,5), 0)
    (thresh, img_bw) = cv.threshold(blur, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    edges = cv.Sobel(img_bw, cv.CV_8U, 1, 0, 5)
    cv.imwrite('edges.jpg', edges)
    return cv.HoughLinesP(edges, 1, np.pi / 180, 10, min_length, max_gap)
    

def detect_strings(img):
    min_length = 25
    max_gap = 5
    blur = cv.GaussianBlur(img, (5,5), 0)
    (thresh, img_bw) = cv.threshold(blur, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    edges = cv.Sobel(img_bw, cv.CV_8U, 0, 1, 3)
    return cv.HoughLinesP(edges, 1, np.pi / 180, 10, min_length, max_gap)


def draw_lines(img, lines, frets=True):
    for x in range(len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img