import tkinter
from tkinter import messagebox
from tkinter import filedialog


def read_doc2vec_model():
    root = tkinter.Tk()
    root.withdraw()

    fTyp = [('', '*.model')]
    iDir = '/home/'

    filename = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    return filename
