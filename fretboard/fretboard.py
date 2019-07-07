import cv2 as cv
import numpy as np 
from neck import Neck, detect_frets, detect_strings, draw_lines
import math

img = cv.imread('guitar-fretboard-2.jpg')

neck = Neck(img).get()
gray = cv.cvtColor(neck, cv.COLOR_BGR2GRAY)

frets = detect_frets(gray)
strings = detect_strings(gray)

neck = draw_lines(neck, frets)
neck = draw_lines(neck, strings)

cv.imwrite('houghlines3.jpg', neck)