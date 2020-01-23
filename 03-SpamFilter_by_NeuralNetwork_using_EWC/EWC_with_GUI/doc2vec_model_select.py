import os
import sys
import tkinter
import EWC_spamfilter_with_GUI
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog


def read_doc2vec():
    fTyp = [('', '*.model')]
    iDir = '/home/'
    filepath = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)


def start_learning(file1):
    file_paths = []
    mail_types = []
    doc2vec_borders = []

    EWC_spamfilter_with_GUI.start(
        file_paths, mail_types, doc2vec_borders)


def main():

    folder_path = tkinter.StringVar()

    # メインウィンドウ
    main_win = tkinter.Tk()
    main_win.title("Main Window")
    main_win.geometry("500x320")

    # メインフレーム
    main_frm = ttk.Frame(main_win)
    main_frm.grid(column=0, row=0, sticky=tkinter.NSEW, padx=0, pady=0)

    # ウィジェット作成（doc2vecファイルのパス）
    folder_label = ttk.Label(main_frm, text="ファイル指定")
    folder_box = ttk.Entry(main_frm, textvariable=folder_path)
    folder_btn = ttk.Button(main_frm, text="参照", command=read_doc2vec)

    # ウィジェット作成（実行ボタン）
    app_btn = ttk.Button(main_frm, text="実行", command=start_learning)

    # ウィジェットの配置
    folder_label.grid(column=0, row=0, pady=10)
    folder_box.grid(column=1, row=0, sticky=tkinter.EW, padx=5)
    folder_btn.grid(column=2, row=0)
    app_btn.grid(column=1, row=2)

    # 配置設定
    main_win.columnconfigure(0, weight=1)
    main_win.rowconfigure(0, weight=1)
    main_frm.columnconfigure(1, weight=1)

    main_win.mainloop()


if __name__ == "__main__":
    main()
