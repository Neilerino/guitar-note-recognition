import cv2 as cv
from finger_tip_finder import get_fingertip_coord

im = cv.imread('test_imgs/key_fing3.jpg')
finger_tip_coords = get_fingertip_coord(im)

cv.circle(im, finger_tip_coords, 3, (0, 0, 255), 3)

cv.imshow('fingers', im)
cv.waitKey(0)
cv.destroyAllWindows()
