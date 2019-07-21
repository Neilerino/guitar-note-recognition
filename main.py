import tkinter as tk
from tkinter import filedialog, Button, Label
import cv2 as cv
from fingertip_detector import get_fingertip_coords


def find_fingertip():
    file_path = filedialog.askopenfilename()
    im = cv.imread(file_path)
    finger_tip_coords = get_fingertip_coords(im)
    for pair in finger_tip_coords:
        cv.circle(im, pair, 3, (0, 0, 255), 4)
    cv.imshow('Fingertip', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


root = tk.Tk()
root.geometry("200x100")
root.resizable(0, 0)
header = Label(root, text='Note Identifier - v0.2')
header.pack()
finger_btn = Button(root, text='Find Fingertip', command=find_fingertip)
finger_btn.pack()
root.mainloop()



