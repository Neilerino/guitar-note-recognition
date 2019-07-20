import cv2 as cv
import numpy as np


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def hist_from_roi(im):
    # TODO: use the rescale frame to make sure the roi window is a reasonable resolution
    roi = cv.selectROI('Select ROI', im, False)
    cv.destroyWindow('Select ROI')
    hist_im_sel = im[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    hist_im_sel = cv.cvtColor(hist_im_sel, cv.COLOR_BGR2HSV)
    hand_hist = cv.calcHist([hist_im_sel], [0, 1], None, (180, 256), [0, 180, 0, 256])
    hand_hist = cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)
    return hand_hist


def get_hist_object(im, hist):
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    back_proj = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    se1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    # FIXME: if returned image is not good often, increase radius of se2
    se2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    back_proj = cv.morphologyEx(back_proj, cv.MORPH_OPEN, se1)
    back_proj = cv.filter2D(back_proj, -1, se2, back_proj)
    back_proj = cv.morphologyEx(back_proj, cv.MORPH_CLOSE, se2)
    ret, thresh = cv.threshold(back_proj, 127, 255, cv.THRESH_BINARY)
    thresh = cv.merge((thresh, thresh, thresh))
    return cv.bitwise_and(im, thresh)


def get_contours(hist_mask_image):
    gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
    _, thresh_im = cv.threshold(gray_hist_mask_image, 0, 255, 0)
    contours, _ = cv.findContours(thresh_im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]
        contour_area = cv.contourArea(cnt)
        if contour_area > max_area:
            max_area = contour_area
            max_i = i

    return contour_list[max_i]


def centroid(max_contour):
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)
        dist = cv.sqrt(cv.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def get_fingertip_coord(im):
    hand_hist = hist_from_roi(im)
    hist_masked_image = get_hist_object(im, hand_hist)
    contour_list = get_contours(hist_masked_image)
    max_cont = max_contour(contour_list)
    contour_centroid = centroid(max_cont)

    if max_cont is not None:
        hull = cv.convexHull(max_cont, returnPoints=False)
        defects = cv.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, contour_centroid)
        return far_point
    else:
        return None