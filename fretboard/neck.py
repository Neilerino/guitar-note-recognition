import cv2 as cv
from math import inf
from statistics import median
import numpy as np

class Neck:

    def __init__(self, img):
        self.neck = crop_neck(img)

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

    return img[first_y - 50: last_y+50]

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

def detect_frets(img, will_blur=True):
    min_length = 4
    max_gap = 0
    if will_blur:
        img = cv.GaussianBlur(img, (5,5), 0)
    (thresh, img) = cv.threshold(img, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    edges = cv.Sobel(img, cv.CV_8U, 1, 0, 3)
    cv.imwrite('edges.jpg', edges)
    return cv.HoughLinesP(edges, 1, np.pi / 180, 10, min_length, max_gap)
    

def detect_strings(img, will_blur=True):
    min_length = 12
    max_gap = 2
    if will_blur:
        img = cv.GaussianBlur(img, (5,5), 0)
    (thresh, img) = cv.threshold(img, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    edges = cv.Sobel(img, cv.CV_8U, 0, 1, 3)
    return cv.HoughLinesP(edges, 1, np.pi / 180, 5, min_length, max_gap)


def draw_lines(img, lines, frets=True):
    for x in range(len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

    return img

def fix_horizontal_lines(img):
    kernal = np.ones((1,20), np.uint8)
    d_im = cv.dilate(img, kernal, iterations=2)
    e_im = cv.erode(d_im, kernal, iterations=2)
    return e_im

def fix_vertical_lines(img):
    kernal = np.ones((20,1), np.uint8)
    d_im = cv.dilate(img, kernal, iterations=2)
    e_im = cv.erode(d_im, kernal, iterations=2)
    return e_im


def combine_lines(lines):
    x_space = 5
    y_space = 5
    new_lines = []
    previous_line_end = [0, 0]
    new_line = np.array([0, 0, 0, 0])
    for x in range(len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            if (abs(x1 - previous_line_end[0]) < x_space) and (abs(y1-previous_line_end[1]) < y_space):
                new_line[2] = x1
                new_line[3] = y1
            else:
                new_line[2] = previous_line_end[0]
                new_line[3] = previous_line_end[1]
                new_lines.append(new_line)
                new_line[0] = x1
                new_line[1] = y1
            previous_line_end[0] = x2
            previous_line_end[1] = y2

    return new_lines
