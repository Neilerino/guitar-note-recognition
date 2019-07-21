import cv2 as cv
import numpy as np 
from fretboard.neck import Neck, detect_frets, detect_strings, draw_lines, fix_horizontal_lines, fix_vertical_lines, \
    combine_lines

def get_lines_image(img): 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    frets = detect_frets(gray)
    strings = detect_strings(gray)

    gray_frets = draw_lines(gray, frets)
    # gray_frets = fix_vertical_lines(gray_frets)

    gray_strings = draw_lines(gray, strings)
    gray_strings = fix_horizontal_lines(gray_strings)

    second_frets = detect_frets(gray_frets, False)
    second_strings = detect_strings(gray_strings, False)

    final = draw_lines(gray, second_frets)
    final = draw_lines(gray, second_strings)

    lines = detect_strings(final, False)
    final = draw_lines(gray, lines)
    i = 2
    for j in range(i):
        lines = detect_strings(final, False)
        final = draw_lines(gray, lines)

    image_with_lines = draw_lines(gray, lines)

    return image_with_lines

def get_note(point, image_with_lines):
    # converting tuple into list
    point = [coord for coord in point] 

    point[0] = point[0] - 5
    point[1] = point[1] - 9

    x_coord = point[0] / image_with_lines.shape[1]
    y_coord = point[1] / image_with_lines.shape[0]

    if ( 0.37 <= x_coord <= 0.40) and ( 0.21 <= y_coord <= 0.26):
        return('A')

    if (0.68 <= x_coord <= 0.73) and ( 0.3 <= y_coord <= 0.33 ):
        return('B')

    if (0.58 <= x_coord <= 0.66) and (0.32 <= y_coord <= 0.37):
        return('C')

    if (0.44 <= x_coord <= 0.51) and (0.38 <= y_coord <= 0.45):
        return('D')

    if (0.71 <= x_coord <= 0.76) and (0.41 <= y_coord <= 0.49):
        return('E')

    if (0.53 <= x_coord <= 0.6) and (0.42 <= y_coord <= 0.49):
        return('F#')

    if (0.5 <= x_coord <= 0.55) and \
        ( ( 0.53 <= y_coord <= 0.58) or 0.21 <= y_coord <= 0.26):
        return('G')

    return 'Could not identify note'