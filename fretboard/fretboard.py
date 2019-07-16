import cv2 as cv
import numpy as np 
from neck import Neck, detect_frets, detect_strings, draw_lines, fix_horizontal_lines, fix_vertical_lines
import math

img = cv.imread('guitar-fretboard-2.jpg')

neck = Neck(img).get()
gray = cv.cvtColor(neck, cv.COLOR_BGR2GRAY)

frets = detect_frets(gray)
strings = detect_strings(gray)

gray_frets = draw_lines(gray, frets)
gray_frets = fix_vertical_lines(gray_frets)

gray_strings = draw_lines(gray, strings)
gray_strings = fix_horizontal_lines(gray_strings)

second_frets = detect_frets(gray_frets)
second_strings = detect_strings(gray_strings)

final = draw_lines(gray, second_frets)
final = draw_lines(final, second_frets)

# print(second_frets)
print(second_strings)

# gray = draw_lines(gray, strings)
# gray = fix_hoirizontal_lines(gray)

cv.imwrite('houghlines3.jpg', final)