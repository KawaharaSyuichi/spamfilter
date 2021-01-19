# 一番普通のEWCを用いたスパムフィルタ
# 2005年のメールを800回学習，2006年のメールを800回学習,2007年のメールを500回学習
import sys
import csv
import gensim
import random
import warnings
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import OrderedDict
from IPython import display
from gensim.models import Doc2Vec
from model_for_doc2vec import Model

random.seed(2021)
tf.compat.v1.set_random_seed(2021)
warnings.filterwarnings(
    'ignore', category=matplotlib.MatplotlibDeprecationWarning)


class parameter:
    """
    ニューラルネットワークで使用するパラメータ
    """

    def __init__(self):
        self.ITERATIONS = 2120
        self.DISP_FREQ = 20
        self.lams = [0]

    def set_lams(self, lams_value):
        self.lams.append(lams_value)


def random_batch(trainset, batch_size, start_idx):
    """
    学習用のバッチ作成
    """
    half_batch_size = int(batch_size / 2)
    return_flag = False

    batch = []
    docvec = []
    docflag = []
    idx = []

    spam_half_start = 100 + start_idx * half_batch_size
    spam_half_end = 100 + (start_idx + 1) * half_batch_size
    ham_half_start = 1100 + start_idx * half_batch_size
    ham_half_end = 1100 + (start_idx + 1) * half_batch_size

    if spam_half_end > 500:  # 学習用データセットを一通り学習した場合(エポック数が1増える)
        idx = [num for num in range(spam_half_start, 500)]
        idx.extend(random.sample(range(100, spam_half_start),
                                 k=half_batch_size - len(idx)))
        idx_1 = [num for num in range(ham_half_start, 1500)]
        idx_1.extend(random.sample(range(1100, ham_half_start),
                                   k=half_batch_size - len(idx_1)))

        # 学習用データセットをシャッフル
        spam_half = trainset[100:500]
        ham_half = trainset[1100:1500]
        random.shuffle(spam_half)
        random.shuffle(ham_half)
        trainset[100:500] = spam_half
        trainset[1100:1500] = ham_half

        return_flag = True
    else:
        idx = [num for num in range(spam_half_start, spam_half_end)]
        random.shuffle(idx)
        idx_1 = [num for num in range(ham_half_start, ham_half_end)]
        random.shuffle(idx_1)
    idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        # ラベル作成
        if num < 1000:
            docflag.append([0.0, 1.0])  # spam
        else:
            docflag.append([1.0, 0.0])  # ham

    batch.append(docvec)
    batch.append(docflag)

    return batch, return_flag


def make_test_batch(trainset):
    """
    テスト用のバッチ作成
    """
    batch = []
    docvec = []
    docflag = []

    # テスト用バッチでは学習用で用いなかった残りのデータを全て使用する
    idx = [num for num in range(500, 1000)]  # スパムメール分
    idx_1 = [num for num in range(1500, 2000)]  # 正規メール分
    idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num < 1000:
            docflag.append([0.0, 1.0])  # spam
        else:
            docflag.append([1.0, 0.0])  # ham

    batch.append(docvec)
    batch.append(docflag)

    return batch


def make_train_batch(trainset):
    """
    全ての学習用データのバッチ作成
    """
    batch = []
    docvec = []
    docflag = []

    # テスト用バッチでは学習用で用いなかった残りのデータを全て使用する
    idx = [num for num in range(100, 500)]  # スパムメール分
    idx_1 = [num for num in range(1100, 1500)]  # 正規メール分
    idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num < 1000:
            docflag.append([0.0, 1.0])  # spam
        else:
            docflag.append([1.0, 0.0])  # ham

    batch.append(docvec)
    batch.append(docflag)

    return batch


def train_new_task(model, num_iter, disp_freq, mail_doc2vec, mail_class, sess, lams, trainsets, mode="SGD"):
    train_start_idx = 0
    train_ephoc = 0
    model.restore(sess)  # 重みとバイアスのパラメータの読み込み

    model.set_vanilla_loss()  # 確率的勾配降下法を用いる

    # テスト用データの識別率を格納するためのリスト作成
    test_accs = []
    for task in range(len(trainsets)):
        test_accs.append(np.zeros(int(num_iter / disp_freq)))

    # 学習用データの識別率を格納するためのリスト作成
    train_accs = []
    for task in range(len(trainsets)):
        train_accs.append(np.zeros(int(num_iter / disp_freq)))

    for iter in range(num_iter):
        if iter % 100 == 0:  # 100回おきにiterをプロット
            print("iteration : ", iter)

        if iter < 800:
            train_batch, return_flag = random_batch(
                trainsets[0], 100, train_start_idx)  # 2005年のメールデータを学習に使用
        elif iter >= 800 and iter < 1600:
            if iter == 800 and mode == "EWC":
                # 各ラベルのフィッシャー情報量の計算
                model.compute_fisher(trainsets[0], sess,
                                     num_samples=200, plot_diffs=False)
                model.stor()  # 重みとバイアスのパラメータ保存
                model.restore(sess)  # 重みとバイアスのパラメータの読み込み
                # Elastic Weight Consolidationを用いる
                model.update_ewc_loss(lams[1])

            train_batch, return_flag = random_batch(
                trainsets[1], 100, train_start_idx)  # 2006年のメールデータを学習に使用
        else:
            if iter == 1600 and mode == "EWC":
                # 各ラベルのフィッシャー情報量の計算
                model.compute_fisher(trainsets[1], sess,
                                     num_samples=200, plot_diffs=False)
                # 重みとバイアスのパラメータ保存
                model.stor()
                model.restore(sess)
                # Elastic Weight Consolidationを用いる
                model.update_ewc_loss(lams[1])

            train_batch, return_flag = random_batch(
                trainsets[2], 100, train_start_idx)  # 2007年のメールデータを学習に使用

        if return_flag == False:
            train_start_idx += 1
        elif return_flag == True:
            train_start_idx = 0
            train_ephoc += 1
            #print("train ephoc is {}".format(train_ephoc))

        # 学習開始
        model.train_step.run(feed_dict={mail_doc2vec: np.array(
            train_batch[0]), mail_class: np.array(train_batch[1])})

        if iter % 20 == 0:  # 20回ごとに識別率を計算
            if iter < 800:  # 2005年のメールに対する識別率のみを計算
                test_accs[1][int(iter / disp_freq)] = None
                test_accs[2][int(iter / disp_freq)] = None
                train_accs[1][int(iter / disp_freq)] = None
                train_accs[2][int(iter / disp_freq)] = None

                # 2005年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[0])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[0][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2005年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[0])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[0][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)
            elif iter >= 800 and iter < 1600:
                test_accs[2][int(iter / disp_freq)] = None
                train_accs[2][int(iter / disp_freq)] = None

                # 2005年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[0])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[0][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2005年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[0])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[0][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)

                # 2006年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[1])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[1][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2006年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[1])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[1][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)
            else:
                # 2005年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[0])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[0][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2005年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[0])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[0][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)

                # 2006年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[1])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[1][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2006年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[1])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[1][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)

                # 2007年のテスト用データに対する識別率を計算
                test_batch = make_test_batch(trainsets[2])
                feed_dict = {mail_doc2vec: np.array(
                    test_batch[0]), mail_class: np.array(test_batch[1])}
                test_accs[2][int(iter / disp_freq)
                             ] = model.accuracy.eval(feed_dict=feed_dict)

                # 2007年の学習用データに対する識別率を計算
                train_all_batch = make_train_batch(trainsets[2])
                feed_train_dict = {mail_doc2vec: np.array(
                    train_all_batch[0]), mail_class: np.array(train_all_batch[1])}
                train_accs[2][int(iter / disp_freq)
                              ] = model.accuracy.eval(feed_dict=feed_train_dict)

    ########################################################
    #                 test 識別率のプロット(個別)             #
    ########################################################
    x_max = 2101
    x_iter = list(range(1, num_iter, disp_freq))
    # テスト用データの学習結果をプロット
    plt.plot(x_iter, test_accs[0], marker='.',
             label='2005')  # 2005年分の識別率をプロット
    plt.plot(x_iter, test_accs[1], marker='.',
             label='2006')  # 2006年分の識別率をプロット
    plt.plot(x_iter, test_accs[2], marker='.',
             label='2007')  # 2007年分の識別率をプロット

    plt.grid(which='major', color='black', linestyle='--')
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(np.arange(0, 2101, 100))
    plt.ylim(bottom=0.40, top=1.01)
    if mode == 'SGD':
        plt.title("SGD [test accuracy]")
    else:
        plt.title("EWC λ:" + str(lams[1]) + " [test accuracy]")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

    ########################################################
    #                train 識別率のプロット(個別)             #
    ########################################################
    # 学習用データの学習結果をプロット
    plt.plot(x_iter, train_accs[0], marker='.',
             label='2005')  # 2005年分の識別率をプロット
    plt.plot(x_iter, train_accs[1], marker='.',
             label='2006')  # 2006年分の識別率をプロット
    plt.plot(x_iter, train_accs[2], marker='.',
             label='2007')  # 2007年分の識別率をプロット

    plt.grid(which='major', color='black', linestyle='--')
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(np.arange(0, 2101, 100))
    plt.ylim(bottom=0.40, top=1.01)
    if mode == 'SGD':
        plt.title("SGD [train accuracy]")
    else:
        plt.title("EWC λ:" + str(lams[1]) + " [train accuracy]")
    plt.xlabel("Iterations")
    plt.ylabel("Train Accuracy")
    plt.show()

    # テスト用と学習用の識別率を表示
    print("2005 tset acc : {:.3f}".format(test_accs[0][-1]))
    print("2006 tset acc : {:.3f}".format(test_accs[1][-1]))
    print("2007 tset acc : {:.3f}".format(test_accs[2][-1]))
    print("2005 train acc : {:.3f}".format(train_accs[0][-1]))
    print("2006 train acc : {:.3f}".format(train_accs[1][-1]))
    print("2007 train acc : {:.3f}".format(train_accs[2][-1]))


def main():
    """
    read_doc2vec.pyにより実行される関数
    """

    common_path = '/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_year_doc2vec_model/'

    model_info_orderdict = OrderedDict({
        '2005': common_path + '2005_2000_mails_doc2vec_all.csv',
        '2006': common_path + '2006_2000_mails_doc2vec_all.csv',
        '2007': common_path + '2007_2000_mails_doc2vec_all.csv'
    })

    model_doc2vec_dict = dict()
    model_doc2vec_learned_list = []
    doc2vec_labels_list = []
    trainsets = []

    PARAMETERS = parameter()

    for doc2vec_label, file_path in model_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            model_doc2vec_dict[doc2vec_label] = [
                [float(v) for v in row] for row in reader]

    for doc2vec_label, file_path in model_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            trainsets.append([[float(v) for v in row] for row in reader])

    sess = tf.InteractiveSession()

    mail_doc2vec = tf.placeholder(
        tf.float32, shape=[None, 300])  # 元はshape=[None,300]
    mail_class = tf.placeholder(
        tf.float32, shape=[None, 2])  # 教師ラベルの種類　ham:[1,0],spam:[0,1]

    # ニューラルネットワークモデルの構築(モデルの具体的な構成はmodel_for_doc2vec.pyを参照)
    model = Model(mail_doc2vec, mail_class)

    sess.run(tf.global_variables_initializer())

    PARAMETERS.set_lams(200)

    train_new_task(model, PARAMETERS.ITERATIONS, PARAMETERS.DISP_FREQ,
                   mail_doc2vec, mail_class, sess, PARAMETERS.lams, trainsets, mode="SGD")

    """
    for doc2vec_label in model_info_orderdict.keys():
        doc2vec_labels_list.append(doc2vec_label)
        model_doc2vec_learned_list.append(model_doc2vec_dict[doc2vec_label])

        train_task(model, PARAMETERS.ITERATIONS, PARAMETERS.DISP_FREQ,
                   model_doc2vec_dict[doc2vec_label], doc2vec_labels_list, model_doc2vec_learned_list, mail_doc2vec, mail_class, sess, PARAMETERS.lams)

        if doc2vec_label == "2007":
            break

        # 各ラベルのフィッシャー情報量の計算
        model.compute_fisher(model_doc2vec_dict[doc2vec_label], sess,
                             num_samples=200, plot_diffs=False)

        # 重みとバイアスのパラメータ保存
        model.stor()

        if len(PARAMETERS.lams) == 1:
            PARAMETERS.set_lams(100)
    """


if __name__ == "__main__":
    main()
