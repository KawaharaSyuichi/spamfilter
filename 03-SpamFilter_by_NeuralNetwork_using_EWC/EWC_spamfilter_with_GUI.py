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


class Train_and_Test:
    def __init__(self, doc2vec_info_list, nn_model_info):
        self.nn_model_info = nn_model_info
        self.doc2vec_info_list = doc2vec_info_list
        self.lams = [0]
        self.num_iter = 320
        self.batch_size = 40
        self.disp_freq = 20
        self.test_accs = np.zeros(int(self.num_iter/self.disp_freq))

    def plot_test_acc(self, plot_handles):
        plt.legend(handles=plot_handles, loc="lower right")
        plt.xlabel("Iterations")
        plt.ylabel("Test Accuracy")
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xticks(np.arange(0, 310, 50))
        plt.ylim(0.0, 1.01)
        plt.grid(which='major', color='black', linestyle='-')
        display.display(plt.gcf())
        display.clear_output(wait=True)

    def set_test_accs(self):
        self.test_accs = np.zeros(int(self.num_iter/self.disp_freq))

    def random_batch(self, mail_vector):
        train_batch = list()
        vector_batch = list()
        type_batch = list()

        vector_batch.extend(mail_vector.spam_vector[:self.batch_size])
        vector_batch.extend(mail_vector.ham_vector[:self.batch_size])

        train_batch.append(vector_batch)

        type_batch.extend([[0.0, 1.0] for _ in range(self.batch_size)])  # spam

        type_batch.extend([[1.0, 0.0] for _ in range(self.batch_size)])  # ham

        return train_batch

    def test(self, iter):
        test_batch = self.random_batch(self.doc2vec_info_list[0])

        test_feed_dict = {self.nn_model_info.mail_doc2vec: np.array(
            test_batch[0]), self.nn_model_info.mail_class: np.array(test_batch[1])}

        self.test_accs[int(iter/self.disp_freq)
                       ] = self.nn_model_info.nn_model.accuracy.eval(feed_dict=test_feed_dict)

        plt.plot(range(1, iter + 2, self.disp_freq), self.test_accs)
        plt.show()

    def train(self):
        for l in range(len(self.lams)):
            self.nn_model_info.nn_model.restore(self.nn_model_info.sess)

            if (self.lams[l] == 0):
                self.nn_model_info.nn_model.set_vanilla_loss()
            else:
                self.nn_model_info.nn_model.update_ewc_loss(self.lams[l])

            self.set_test_accs()

            for iter in range(self.num_iter):
                train_batch = self.random_batch(
                    self.doc2vec_info_list[0])

                train_feed_dict = {self.nn_model_info.mail_doc2vec: np.array(
                    train_batch[0]), self.nn_model_info.mail_class: np.array(train_batch[1])}

                self.nn_model_info.nn_model.train_step.run(
                    feed_dict=train_feed_dict)

                if iter % self.disp_freq == 0:
                    self.test(iter)


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

    def set_border(self, border):
        self.border = border

    def get_vector_size(self):
        return len(self.spam_vector)


class NeuralNetworkInformation:
    def __init__(self, VECTOR_SIZE):  # current mail_doc2vec_vector_size : 300
        self.sess = tf.InteractiveSession()
        self.mail_doc2vec = tf.placeholder(
            tf.float32, shape=[None, VECTOR_SIZE])
        self.mail_class = tf.placeholder(
            tf.float32, shape=[None, 2])  # ham:[1,0] spam:[0,1]
        self.nn_model = Model(self.mail_doc2vec, self.mail_class)

    def sess_run(self):
        self.sess.run(tf.global_variables_initializer())


def read_doc2vec(doc2vec_info_list):
    while True:
        doc2vec_info = Doc2vecInformation()
        doc2vec_info.set_file_path(doc2vec_model_select.read_doc2vec_model())

        # 入力するdoc2vecがどの言語や，どの月のものかを入力
        mail_type = input('Input mail type >>')
        doc2vec_info.set_mail_type(mail_type)

        # doc2vecのどこまでがspamで，どこからがhamメールかを入力
        doc2vec_border = input('Input doc2vec border >>')
        doc2vec_info.set_border(int(doc2vec_border))

        doc2vec_info.separata_spam_and_ham(
            Doc2Vec.load(doc2vec_info.file_path))

        doc2vec_info_list.append(doc2vec_info)

        # 読み込みを終わる場合はfを入力
        finish_flag = input('Push f to finish >>')
        if finish_flag == 'f' or finish_flag == 'F':
            break


doc2vec_info_list = []
read_doc2vec(doc2vec_info_list)

nn_model_info = NeuralNetworkInformation(
    doc2vec_info_list[0].get_vector_size())
nn_model_info.sess_run()

Train_and_Test(doc2vec_info_list, nn_model_info).train()
