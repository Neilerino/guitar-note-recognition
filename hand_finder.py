import cv2 as cv


def hand_histogram(im):
    roi = cv.selectROI('hand', im, False)
    hist_im_sel = im[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    hist_im_sel = cv.cvtColor(hist_im_sel, cv.COLOR_BGR2HSV)
    hand_hist = cv.calcHist([hist_im_sel], [0, 1], None, (180, 256), [0, 180, 0, 256])
    hand_hist = cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)
    return hand_hist


def find_hand(hist_ref, im):

    hand_hist = hand_histogram(im)
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    back_proj = cv.calcBackProject([im_hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)

    # cv.imshow('back', back_proj)

    se1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    se2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    # opened = cv.morphologyEx(opened, cv.MORPH_DILATE, se2)
    back_proj = cv.morphologyEx(back_proj, cv.MORPH_OPEN, se1)

    # cv.imshow('de-noise', back_proj)

    back_proj = cv.filter2D(back_proj, -1, se2, back_proj)
    back_proj = cv.morphologyEx(back_proj, cv.MORPH_CLOSE, se2)

    # cv.imshow('open', back_proj)

    ret, thresh = cv.threshold(back_proj, 127, 255, cv.THRESH_BINARY)
    # thresh = cv.merge((thresh, thresh, thresh))
    # return cv.bitwise_and(im, thresh)
    return thresh


# if __name__ == '__main__':
# im = cv.imread('key_fing2.jpg')
# hist_ref_im = cv.imread('hand_for_hist.jpg')
# thresh_im = find_hand(hist_ref_im, hist_ref_im)
#
# cv.imshow('thresh', thresh_im)
# cv.waitKey(0)
# cv.destroyAllWindows()
