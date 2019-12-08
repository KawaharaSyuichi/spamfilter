import random
import sys
import doc2vec_model_select  # 自作のライブラリ
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
from model_for_doc2vec import Model  # 自作のライブラリ
from gensim.models import Doc2Vec


def cal_time(func):  # 実行時間の計測
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("Execution time{:.3f}".format(end - start), "s")
    return wrapper


class Doc2vecInformation:
    def __init__(self):
        self.file_path = str()
        self.mail_type = str()
        self.spam_vector = list()
        self.ham_vector = list()
        self.border = 0

    def separata_spam_and_ham(self, model):
        vector = [model.docvecs[i] for i in range(len(model.docvecs))]
        self.spam_vector = vector[:self.border]
        self.ham_vector = vector[self.border:]

    def set_file_path(self, path):
        self.file_path = path

    def set_mail_type(self, mail_type):
        self.mail_type = mail_type


class NeuralNetworkInformation:
    def __init__(self, mail_doc2vec_vector_size):  # current mail_doc2vec_vector_size : 300
        self.sess = tf.InteractiveSession()
        self.mail_doc2vec = tf.placeholder(
            tf.float32, shape=[None, mail_doc2vec_vector_size])
        self.mail_class = tf.placeholder(
            tf.float32, shape=[None, 2])  # ham:[1,0] spam:[0,1]
        self.nn_model = Model(self.mail_doc2vec, self.mail_class)

    def sess_run(self):
        self.sess.run(tf.global_variables_initializer())


def read_doc2vec():
    while True:
        doc2vec_info = Doc2vecInformation()
        doc2vec_info.set_file_path(doc2vec_model_select.read_doc2vec_model())

        # 入力するdoc2vecがどの言語や，どの月のものかを入力
        mail_type = input('Input mail type >>')
        doc2vec_info.set_mail_type(mail_type)

        # doc2vecのどこまでがspamで，どこからがhamメールかを入力
        doc2vec_border = input('Input doc2vec border >>')
        doc2vec_info.border = int(doc2vec_border)

        doc2vec_info.separata_spam_and_ham(
            Doc2Vec.load(doc2vec_info.file_path))

        doc2vec_info_list.append(doc2vec_info)

        # 読み込みを終わる場合はfを入力
        finish_flag = input('Push f to finish >>')
        if finish_flag == 'f' or finish_flag == 'F':
            break


doc2vec_info_list = []
read_doc2vec()
