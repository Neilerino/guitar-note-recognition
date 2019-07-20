import tkinter as tk
from tkinter import filedialog, Button, Label
import pickle
import cv2 as cv
from finger_tip_finder import get_fingertip_coord, hist_from_roi


def make_profile_from_cam():
    cam = cv.VideoCapture(0)
    frame = None
    print("Take a picture of your hand w/ space")
    while cv.waitKey(33) != 32:
        frame = cam.read()[1]
        cv.imshow('Take a picture of your hand', frame)
    cv.destroyWindow('Take a picture of your hand')
    hist = hist_from_roi(frame)
    with open('hist_profile.pickle', 'wb') as file:
        pickle.dump(hist, file)


def make_profile_from_pic():
    file_path = filedialog.askopenfilename()
    im = cv.imread(file_path)
    hist = hist_from_roi(im)
    with open('hist_profile.pickle', 'wb') as file:
        pickle.dump(hist, file)


def find_finger():
    with open('hist_profile.pickle', 'rb') as file:
        hist = pickle.load(file)

    file_path = filedialog.askopenfilename()
    im = cv.imread(file_path)
    finger_tip_coords = get_fingertip_coord(im, hist)

    cv.circle(im, finger_tip_coords, 3, (0, 0, 255), 4)
    cv.imshow('Fingertip', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


root = tk.Tk()
root.geometry("200x100")
root.resizable(0, 0)
header = Label(root, text='Note Identifier - v0.1')
header.pack()
profile_btn1 = Button(root, text='New profile from pic', command=make_profile_from_pic)
profile_btn1.pack()
profile_btn2 = Button(root, text='New profile from cam', command=make_profile_from_cam)
profile_btn2.pack()
finger_btn = Button(root, text='Find Finger', command=find_finger)
finger_btn.pack()
root.mainloop()



