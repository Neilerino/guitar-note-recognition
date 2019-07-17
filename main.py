import cv2 as cv
import hand_finder as hf

im = cv.imread('test_imgs/fingers.jpg')

hand = hf.find_hand(im, im)
contours, _ = cv.findContours(hand, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hull_ind = []
hull_ctr = []
for i in range(len(contours)):
    hull_ind.append(cv.convexHull(contours[i], returnPoints=False))
    hull_ctr.append(cv.convexHull(contours[i]))

defects = []
for i in range(len(contours)):
    defects.append(cv.convexityDefects(contours[i], hull_ind[i]))

cv.drawContours(im, contours, -1, (0, 255, 0), 3)
cv.drawContours(im, hull_ctr, -1, (0, 0, 255), 3)

cv.imshow('fingers', im)
cv.waitKey(0)
cv.destroyAllWindows()

