import gensim
import random
import sys
import doc2vec_model_select  # 自作のライブラリ
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
from model_for_doc2vec import Model  # 自作のライブラリ
from gensim.models import Doc2Vec


class Doc2vecInformation:
    def __init__(self):
        self.doc2vec_model_file_name = str()
        self.doc2vec_model_type = str()
        self.doc2vec_border = 0
        self.spam_doc2vec = list()
        self.ham_doc2vec = list()

    def separata_spam_and_ham(self, doc2vec_model):
        doc2vec_vector = [doc2vec_model.docvecs[num]
                          for num in range(len(doc2vec_model.docvecs))]
        self.spam_doc2vec = doc2vec_vector[:self.doc2vec_border]
        self.ham_doc2vec = doc2vec_vector[self.doc2vec_border:]


class Doc2vecTrainAndTest:
    def __init__(self, neural_network_model, sess, doc2vec_info_list, mail_doc2vec, mail_class):
        self.sess = sess
        self.mail_doc2vec = mail_doc2vec
        self.mail_class = mail_class
        self.num_iter = 320
        self.disp_freq = 20
        self.batch_size = 80
        self.lams = [0]
        self.learning_doc2vec_list = list()
        self.neural_network_model = neural_network_model
        self.doc2vec_info_list = doc2vec_info_list

    def add_lams_list(self, lam):
        self.lams.append(lam)

    def add_learning_doc2vec_list(self, doc2vec_vector):
        self.learning_doc2vec_list.append(doc2vec_vector)

    def random_batch(self, doc2vec_vector):
        batch_vector = doc2vec_vector[:self.batch_size]
        return batch_vector

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

    # spamとhamの識別率を別々に計算
    def train_task(self):
        for l in range(len(self.lams)):
            self.neural_network_model.restore(self.sess)

            if (self.lams[l] == 0):
                self.neural_network_model.set_vanilla_loss()
            else:
                self.neural_network_model.update_ewc_loss(self.lams[l])

            test_accs = []
            for _ in range(len(self.learning_doc2vec_list)):
                test_accs.append(np.zeros(int(self.num_iter / self.disp_freq)))

            for iter in range(self.num_iter):

                if len(self.lams) == 2:
                    spam_train_batch = self.random_batch(
                        self.learning_doc2vec_list[-1])
                    ham_train_batch = self.random_batch(
                        self.learning_doc2vec_list[-1])

                    train_vector_batch = spam_train_batch[0] + \
                        ham_train_batch[0]
                    train_mail_class_batch = spam_train_batch[1] + \
                        ham_train_batch[1]

                    self.neural_network_model.train_step.run(
                        feed_dict={self.mail_doc2vec: np.array(train_vector_batch), self.mail_class: np.array(train_mail_class_batch)})

                if iter % self.disp_freq == 0:
                    plt.subplot(1, len(self.lams), l + 1)

                    plots = []
                    colors = ['r', 'b', 'c']

                    for task in range(self.learning_doc2vec_list):
                        spam_test_batch = self.random_batch(
                            self.learning_doc2vec_list[-1])
                        ham_test_batch = self.random_batch(
                            self.learning_doc2vec_list[-1])

                        test_vector_batch = spam_test_batch[0] + \
                            ham_test_batch[0]
                        test_mail_class_batch = spam_test_batch[1] + \
                            ham_test_batch[1]

                        feed_dict = {self.mail_class: np.array(
                            test_vector_batch), self.mail_class: np.array(test_mail_class_batch)}

                        test_accs[task][int(
                            iter / self.disp_freq)] = self.neural_network_model.accuracy.eval(feed_dict=feed_dict)

                        if task % 2 == 0:
                            c = "まだ考え中"

                        if l == 0:
                            print("SGD" + " " + c + ":" +
                                  str(test_accs[task][int(iter / self.disp_freq)]))
                        else:
                            print("EWC" + " " + c + ":" +
                                  str(test_accs[task][int(iter / self.disp_freq)]))

                        plot_h, = plt.plot(range(
                            1, iter + 2, self.disp_freq), test_accs[task][:int(iter / self.disp_freq) + 1], colors[task], label=c)

                        plots.append(plot_h)

                    self.plot_test_acc(plots)

                    if l == 0:
                        plt.title("Stochastic Gradient Descent")
                    else:
                        plt.title("Elastic Weight Consolidation")

                    plt.gcf().set_size_inches(len(self.lams) * 5, 3.5)

        plt.show()


def initialization():
    sess = tf.InteractiveSession()
    mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
    mail_class = tf.placeholder(
        tf.float32, shape=[None, 2])  # ham:[1,0],spam:[0,1]

    neural_network_model = Model(mail_doc2vec, mail_class)

    sess.run(tf.global_variables_initializer())

    return neural_network_model, sess, mail_doc2vec, mail_class


# doc2vec[1000]:spam
# doc2vec[1000:2000]:ham

doc2vec_info_list = []
while True:
    doc2vec_info = Doc2vecInformation()
    doc2vec_info.doc2vec_model_file_name = doc2vec_model_select.read_doc2vec_model()

    # 入力するdoc2vecがどの言語や，どの月のものかを入力
    doc2vec_type = input('Input doc2vec type >>')
    doc2vec_info.doc2vec_model_type = doc2vec_type

    # doc2vecのどこまでがspamで，どこからがhamメールかを入力
    doc2vec_border = input('Input doc2vec border >>')
    doc2vec_info.doc2vec_border = int(doc2vec_border)

    doc2vec_info.separata_spam_and_ham(
        Doc2Vec.load(doc2vec_info.doc2vec_model_file_name))

    doc2vec_info_list.append(doc2vec_info)

    # 読み込みを終わる場合はfを入力
    finish_flag = input('Push f to finish >>')
    if finish_flag == 'f' or finish_flag == 'F':
        break

neural_network_model, sess, mail_doc2vec, mail_class = initialization()

doc2vec_train_and_test = Doc2vecTrainAndTest(
    neural_network_model, sess, doc2vec_info_list, mail_doc2vec, mail_class)

for i, doc2vec_info in enumerate(doc2vec_info_list):
    doc2vec_train_and_test.add_learning_doc2vec_list(
        doc2vec_info_list[i*2])  # spamを追加
    doc2vec_train_and_test.add_learning_doc2vec_list(
        doc2vec_info_list[i*2+1])  # hamを追加
