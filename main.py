import tkinter as tk
from tkinter import filedialog, Button, Label
import cv2 as cv
from fingertip_detector import get_fingertip_coords
from fretboard.neck import Neck
from fretboard.fretboard import get_note, get_lines_image

def find_fingertip(file_path=None):
    if not file_path:
        file_path = filedialog.askopenfilename()
    im = cv.imread(file_path)
    im = Neck(im).get()
    finger_tip_coords = get_fingertip_coords(im)
    i_w_lines = get_lines_image(im)
    for pair in finger_tip_coords:
        cv.circle(im, pair, 3, (0, 0, 255), 4)
        note = get_note(pair, i_w_lines)
        cv.putText(im, note, (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        print('Note: ', note)
    cv.imshow('Fingertip', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


root = tk.Tk()
root.geometry("150x50")
root.resizable(0, 0)
header = Label(root, text='Note Identifier - v1.0')
header.pack()
finger_btn = Button(root, text='Get note', command=find_fingertip)
finger_btn.pack()
root.mainloop()



