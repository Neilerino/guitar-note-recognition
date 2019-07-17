import cv2 as cv


im = cv.imread('test_imgs/fingers.jpg')
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)


img_blur = cv.GaussianBlur(img, (0, 0), 1.5, 1.5)
im_edges = cv.Canny(img_blur, 100, 200)
_, im_edges = cv.threshold(im_edges, 127, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(im_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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

