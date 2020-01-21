import os
import sys
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog


def button1_clicked(file1):
    fTyp = [('', '*.model')]
    iDir = '/home/'
    filepath = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    file1.set(filepath)


def button2_clicked(file1):
    messagebox.showinfo('FileReference Tool', u'参照ファイルは↓↓\n' + file1.get())


def read_doc2vec_model():
    root = Tk()
    root.title("Main Menu")

    # Frame1の作成
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid()

    # 「ファイル>>」ラベルの作成
    s = StringVar()
    s.set('ファイル>>')
    label1 = ttk.Label(frame1, textvariable=s)
    label1.grid(row=0, column=0)

    # 参照ファイルパス表示ラベルの作成
    file1 = StringVar()
    file1_entry = ttk.Entry(frame1, textvariable=file1, width=50)
    file1_entry.grid(row=0, column=2)

    # 参照ボタンの作成
    button1 = ttk.Button(frame1, text='参照', command=button1_clicked(file1))
    button1.grid(row=0, column=3)

    # Frame2の作成
    frame2 = ttk.Frame(root, padding=(0, 5))
    frame2.grid(row=1)

    # Startボタンの作成
    button2 = ttk.Button(frame2, text='Start', command=button2_clicked(file1))
    button2.pack(side=LEFT)

    # Cancelボタンの作成
    button3 = ttk.Button(frame2, text='Cancel', command=quit)
    button3.pack(side=LEFT)

    root.mainloop()

    # return filename
