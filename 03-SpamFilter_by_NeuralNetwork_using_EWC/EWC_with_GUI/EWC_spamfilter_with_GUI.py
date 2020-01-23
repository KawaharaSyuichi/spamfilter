from gensim.models import Doc2Vec
from model_for_doc2vec_with_GUI import Model  # 自作のライブラリ
from IPython import display

import random
# import doc2vec_model_select  # 自作のライブラリ
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
        self.lams = [0, 40]
        self.num_iter = 320
        self.disp_freq = 20
        self.batch_size = 40  # spamとhamで，合計80バッチ
        self.test_accs = np.zeros(int(self.num_iter / self.disp_freq))
        self.iteration = 0
        self.learned_doc2vec_list = list()
        self.plots = list()

    def plot_test_acc(self, plot_handles):
        plt.legend(handles=plot_handles, loc="best")
        plt.xlabel("Iterations")
        plt.ylabel("Test Accuracy")
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xticks(np.arange(0, 310, 50))
        plt.ylim(0.0, 1.01)
        plt.grid(which='major', color='black', linestyle='-')
        display.display(plt.gcf())
        display.clear_output(wait=True)

    def set_test_accs(self, doc2vec_info_id):
        self.test_accs = []

        for _ in range(doc2vec_info_id + 1):
            self.test_accs.append(
                np.zeros(int(self.num_iter / self.disp_freq)))

    def iteration_up(self):
        self.iteration += 1

    def iteration_reset(self):
        self.iteration = 0

    def add_learned_doc2vec(self, doc2vec_info):
        self.learned_doc2vec_list.append(doc2vec_info)

    def reset_plots(self):
        self.plots = list()

    def add_plot(self, plot_h):
        self.plots.append(plot_h)

    def test_random_batch(self, spam_mail_vector, ham_mail_vector):
        test_batch = list()
        vector_batch = list()
        type_batch = list()

        vector_batch.extend(spam_mail_vector)
        vector_batch.extend(ham_mail_vector)
        test_batch.append(vector_batch)

        type_batch.extend([[0.0, 1.0]
                           for _ in range(len(spam_mail_vector))])  # spam
        type_batch.extend([[1.0, 0.0]
                           for _ in range(len(ham_mail_vector))])  # ham
        test_batch.append(type_batch)

        return test_batch

    def train_random_batch(self, spam_mail_vector, ham_mail_vector):
        train_batch = list()
        vector_batch = list()
        type_batch = list()

        if (self.iteration + 1) * self.batch_size >= len(spam_mail_vector):
            vector_batch.extend(
                spam_mail_vector[self.iteration*self.batch_size:])
            vector_batch.extend(
                random.sample(spam_mail_vector[:self.iteration*self.batch_size], self.batch_size - len(spam_mail_vector[self.iteration*self.batch_size:])))

            vector_batch.extend(
                ham_mail_vector[self.iteration*self.batch_size:])
            vector_batch.extend(
                random.sample(ham_mail_vector[:self.iteration*self.batch_size], self.batch_size - len(ham_mail_vector[self.iteration*self.batch_size:])))

            self.iteration_reset()
            random.shuffle(spam_mail_vector)
            random.shuffle(ham_mail_vector)
        else:
            vector_batch.extend(
                spam_mail_vector[self.iteration*self.batch_size:(self.iteration + 1)*self.batch_size])
            vector_batch.extend(
                ham_mail_vector[self.iteration*self.batch_size:(self.iteration + 1)*self.batch_size])

        self.iteration_up()

        train_batch.append(vector_batch)

        type_batch.extend([[0.0, 1.0] for _ in range(self.batch_size)])  # spam
        type_batch.extend([[1.0, 0.0] for _ in range(self.batch_size)])  # ham
        train_batch.append(type_batch)

        return train_batch

    def test(self, iter, l):
        plt.subplot(1, len(self.lams), l + 1)

        for doc2vec_info_id, doc2vec_info in enumerate(self.learned_doc2vec_list):
            test_batch = self.test_random_batch(
                doc2vec_info.test_spam_vector, doc2vec_info.test_ham_vector)

            test_feed_dict = {self.nn_model_info.mail_doc2vec: np.array(
                test_batch[0]), self.nn_model_info.mail_class: np.array(test_batch[1])}

            self.test_accs[doc2vec_info_id][int(
                iter/self.disp_freq)] = self.nn_model_info.nn_model.accuracy.eval(feed_dict=test_feed_dict)

            plot_h,  = plt.plot(range(1, iter + 2, self.disp_freq),
                                self.test_accs[doc2vec_info_id][:int(iter / self.disp_freq) + 1], label=doc2vec_info.mail_type)

            self.add_plot(plot_h)

        self.plot_test_acc(self.plots)

        if l == 0:
            plt.title("Stochastic Gradient Descent")  # 確率的勾配降下法
        else:
            plt.title("Elastic Weight Consolidation")

    @cal_time
    def train(self):
        for doc2vec_info_id, doc2vec_info in enumerate(self.doc2vec_info_list):
            self.add_learned_doc2vec(doc2vec_info)

            for l in range(len(self.lams)):
                """
                学習準備
                """
                self.nn_model_info.nn_model.restore(self.nn_model_info.sess)

                if (self.lams[l] == 0):
                    self.nn_model_info.nn_model.set_vanilla_loss()
                else:
                    if doc2vec_info_id == 0:  # 一つ目のdoc2vecは追加学習しない
                        break

                    self.nn_model_info.nn_model.update_ewc_loss(self.lams[l])

                self.set_test_accs(doc2vec_info_id)

                """
                学習実行
                """
                for iter in range(self.num_iter):
                    train_batch = self.train_random_batch(
                        doc2vec_info.train_spam_vector, doc2vec_info.train_ham_vector)

                    train_feed_dict = {self.nn_model_info.mail_doc2vec: np.array(
                        train_batch[0]), self.nn_model_info.mail_class: np.array(train_batch[1])}

                    self.nn_model_info.nn_model.train_step.run(
                        feed_dict=train_feed_dict)

                    """
                    self.disp_freq回学習するごとにテスト実行
                    """
                    if iter % self.disp_freq == 0:
                        self.test(iter, l)

            plt.savefig(doc2vec_info.mail_type + ".png")

            if doc2vec_info_id + 1 != len(self.doc2vec_info_list):
                self.nn_model_info.nn_model.compute_fisher(
                    doc2vec_info.fisher_vector, self.nn_model_info.sess, num_samples=200)
                self.nn_model_info.nn_model.star()


class Doc2vecInformation:
    def __init__(self):
        self.file_path = str()
        self.mail_type = str()
        self.train_spam_vector = list()
        self.train_ham_vector = list()
        self.test_spam_vector = list()
        self.test_ham_vector = list()
        self.fisher_vector = list()  # フィッシャー情報行列を計算するためのベクトル
        self.border = 0

    def separata_spam_and_ham(self, model):
        vector = [model.docvecs[i] for i in range(len(model.docvecs))]

        self.train_spam_vector = vector[:int(self.border / 2)]
        self.train_ham_vector = vector[self.border:self.border +
                                       int(self.border / 2)]
        self.test_spam_vector = vector[int(self.border / 2):self.border]
        self.test_ham_vector = vector[self.border + int(self.border / 2):]

        self.fisher_vector.extend(random.sample(self.train_spam_vector, 100))
        self.fisher_vector.extend(random.sample(self.train_ham_vector, 100))

    def set_file_path(self, path):
        self.file_path = path

    def set_mail_type(self, mail_type):
        self.mail_type = mail_type

    def set_border(self, border):
        self.border = border

    def get_vector_size(self):
        return len(self.train_spam_vector[0])


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


def read_doc2vec(doc2vec_info_list, doc2vec_file_paths, mail_types, doc2vec_borders):
    for (doc2vec_file_path, mail_type, doc2vec_border) in zip(doc2vec_file_paths, mail_types, doc2vec_borders):
        doc2vec_info = Doc2vecInformation()
        # doc2vecファイルのパスを追加
        doc2vec_info.set_file_path(doc2vec_file_path)

        # 入力するdoc2vecがどの言語や，どの月のものか
        doc2vec_info.set_mail_type(mail_type)

        # doc2vecのどこまでがspamで，どこからがhamメールかを入力
        doc2vec_info.set_border(int(doc2vec_border))

        doc2vec_info.separata_spam_and_ham(
            Doc2Vec.load(doc2vec_info.file_path))

        doc2vec_info_list.append(doc2vec_info)


def start(file_paths, mail_types, doc2vec_borders):
    doc2vec_info_list = []
    read_doc2vec(doc2vec_info_list, file_paths, mail_types, doc2vec_borders)

    nn_model_info = NeuralNetworkInformation(
        doc2vec_info_list[0].get_vector_size())

    nn_model_info.sess_run()

    Train_and_Test(doc2vec_info_list, nn_model_info).train()

# start()
