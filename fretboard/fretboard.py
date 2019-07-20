import cv2 as cv
import numpy as np 
from neck import Neck, detect_frets, detect_strings, draw_lines, fix_horizontal_lines, fix_vertical_lines, \
    combine_lines
import math

img = cv.imread('test-pictures/G.jpg')

neck = Neck(img).get()

# neck = neck[0:neck.shape[1], int(neck.shape[2] * 0.4) : neck.shape[2]]

gray = cv.cvtColor(neck, cv.COLOR_BGR2GRAY)

frets = detect_frets(gray)
strings = detect_strings(gray)

gray_frets = draw_lines(gray, frets)
gray_frets = fix_vertical_lines(gray_frets)

gray_strings = draw_lines(gray, strings)
gray_strings = fix_horizontal_lines(gray_strings)

# second_frets = detect_frets(gray_frets, False)
second_strings = detect_strings(gray_strings, False)

# final = draw_lines(neck, second_frets)
final = draw_lines(gray, second_strings)

lines = detect_strings(final, False)
final = draw_lines(gray, lines)
i = 1
for j in range(i):
    lines = detect_strings(final, False)
    final = draw_lines(gray, lines)

# print(second_frets)
# print(second_strings)

image_with_lines = draw_lines(neck, lines)

# gray = draw_lines(gray, strings)
# gray = fix_hoirizontal_lines(gray)
point = [ 1030, 75 ]
cv.circle(image_with_lines,(point[0], point[1]), 10, (0,0,255), -1)

x_coord = point[0] / image_with_lines.shape[1]
y_coord = point[1] / image_with_lines.shape[0]

if ( 0.37 <= x_coord <= 0.40) and ( 0.21 <= y_coord <= 0.25):
    print('A')

if (0.68 <= x_coord <= 0.71) and ( 0.3 <= y_coord <= 0.33 ):
    print('B')

if (0.58 <= x_coord <= 0.63) and (0.29 <= y_coord <= 0.33):
    print('C')

if (0.44 <= x_coord <= 0.47) and (0.38 <= y_coord <= 0.42):
    print('D')

if (0.71 <= x_coord <= 0.74) and (0.41 <= y_coord <= 0.76):
    print('E')

if (0.53 <= x_coord <= 0.57) and (0.42 <= y_coord <= 0.46):
    print('F#')

if (0.5 <= x_coord <= 0.55) and \
    ( ( 0.53 <= y_coord <= 0.58) or 0.21 <= y_coord <= 0.26):
    print('G')

cv.imwrite('houghlines3.jpg', image_with_lines)