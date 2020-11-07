import sys
import tkinter as tk
import EWC_spamfilter as EWC
import EWC_spamfilter_with_MVG as EWC_MVG
from tkinter import filedialog
from tkinter import ttk
from collections import OrderedDict


class Application(tk.Frame):
    def __init__(self, args, master=None):
        """
        初期設定
        """
        super().__init__(master)
        self.master.geometry()
        self.master.title("EWC")
        self.model_info_orderdict = OrderedDict()
        self.combobox_list = []
        self.combobox = ttk.Combobox(self.master, values=self.combobox_list)

        if args[1] == "mvg":
            self.mvg_flag = True
        else:
            self.mvg_flag = False

        self.create_window()

    def input_file_path(self):
        """
        読み込むファイルを選択
        読み込むファイルの絶対パスを保存
        """
        typ = [("CSVファイル", ".csv")]
        initial_dir = ""
        file_path = filedialog.askopenfilename(
            filetypes=typ, initialdir=initial_dir)

        self.file_path_entry.insert(tk.END, file_path)

    def input_file_label(self):
        """
        読み込むファイルにラベルを付ける
        """
        label = self.file_label_entry.get()

        if label == "":
            return

        self.model_info_orderdict[label] = self.file_path_entry.get()
        self.combobox_list.append(
            str(len(self.combobox_list) + 1) + "：" + label)

        self.combobox["values"] = self.combobox_list

        self.file_path_entry.delete(0, tk.END)
        self.file_label_entry.delete(0, tk.END)

    def run_ewc(self):
        """
        EWC_spamfilter.pyを実行
        """
        if self.mvg_flag == False:
            EWC.main(self.model_info_orderdict)
        else:
            EWC_MVG.main(self.model_info_orderdict)

    def create_window(self):
        """
        GUIの設定
        """
        self.file_path_entry = tk.Entry(self.master, justify="left", width=50)
        self.file_path_entry.grid(row=0, column=0, pady=0, padx=0)

        self.file_label_entry = tk.Entry(self.master, justify="left", width=50)
        self.file_label_entry.grid(row=1, column=0, pady=0, padx=0)

        tk.Button(self.master, text="参照", command=self.input_file_path,
                  width=4).grid(row=0,  column=1)
        tk.Button(self.master, text="ラベル", command=self.input_file_label,
                  width=4).grid(row=1, column=1)

        self.combobox.grid(row=2, column=0)

        tk.Button(self.master, text="実行", command=self.run_ewc, width=10).grid(
            row=3, column=0, columnspan=2)


if __name__ == "__main__":
    args = sys.argv

    if len(args) == 1:
        sys.exit(
            "In case of using MVG, input mvg.\nIn case of not using MVG, input othre string")

    root = tk.Tk()
    app = Application(master=root, args=args)
    app.mainloop()
