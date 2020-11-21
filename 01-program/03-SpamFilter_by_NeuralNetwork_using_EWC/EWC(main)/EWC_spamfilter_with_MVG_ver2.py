"""
2020/11/24日の輪行用のプログラム(詳しいことはOneDriveのPPTを参照)
"""
import sys
import csv
import copy
import gensim
import random
import warnings
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from gensim.models import Doc2Vec
from collections import OrderedDict
from model_for_doc2vec_ver2 import Model

random.seed(2020)
tf.compat.v1.set_random_seed(2020)
warnings.filterwarnings(
    'ignore', category=matplotlib.MatplotlibDeprecationWarning)


class parameter:
    """
    ニューラルネットワークで使用するパラメータ
    """

    def __init__(self):
        self.ITERATIONS = 1220  # 元は520
        self.DISP_FREQ = 20  # 元は20
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
    テスト用のバッチ作成
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


def plot_test_acc(plot_handles):
    """
    学習結果のプロット図を作成
    """

    plt.legend(handles=plot_handles, loc="lower right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.yticks(np.arange(0.45, 1.1, 0.05))
    plt.xticks(np.arange(0, 520, 20))  # 元は0,520,20
    plt.ylim(0.45, 1.01)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def train_task(model, num_iter, disp_freq, trainset, doc2vec_labels, testsets, mail_doc2vec, mail_class, sess, lams):
    """
    学習と識別の実行
    """

    train_start_idx = 0
    train_ephoc = 0
    mvg_train_start_idx = 0
    mvg_train_ephoc = 0

    for l in range(len(lams)):
        first_flag = False

        model.restore(sess)  # 重みとバイアスのパラメータの読み込み

        if(lams[l] == 0):
            model.set_vanilla_loss()  # 確率的勾配降下法を用いる
        else:
            model.update_ewc_loss(lams[l])  # Elastic Weight Consolidationを用いる

        # 識別率を格納するためのリスト作成
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter / disp_freq)))

        # num_iter回学習を行う
        for iter in range(num_iter):
            # 初学習の場合と学習回数が20回以上の場合は新しい学習データでのみ学習
            if iter > 100 or len(testsets) == 1:
                if len(lams) == 2 and first_flag == True:  # 初の学習を行う場合
                    train_batch, return_flag = random_batch(
                        trainset, 32, train_start_idx)

                    if return_flag == False:
                        train_start_idx += 1
                    elif return_flag == True:
                        train_start_idx = 0
                        train_ephoc += 1
                        print("train ephoc is {}".format(train_ephoc))

                    # 学習開始
                    model.train_step.run(feed_dict={mail_doc2vec: np.array(
                        train_batch[0]), mail_class: np.array(train_batch[1])})

                elif len(lams) == 2 and first_flag == False:
                    first_flag = True
                elif len(lams) == 1:
                    train_batch, return_flag = random_batch(
                        trainset, 32, train_start_idx)

                    if return_flag == False:
                        train_start_idx += 1
                    elif return_flag == True:
                        train_start_idx = 0
                        train_ephoc += 1
                        print("train ephoc is {}".format(train_ephoc))

                    # 学習開始
                    model.train_step.run(feed_dict={mail_doc2vec: np.array(
                        train_batch[0]), mail_class: np.array(train_batch[1])})

            elif iter % 2 == 0:  # 学習回数が100回未満、かつ学習回数が偶数回の場合
                if len(lams) == 2 and first_flag == True:  # 初の学習を行う場合
                    train_batch, return_flag = random_batch(
                        trainset, 32, train_start_idx)

                    if return_flag == False:
                        train_start_idx += 1
                    elif return_flag == True:
                        train_start_idx = 0
                        train_ephoc += 1
                        print("train ephoc is {}".format(train_ephoc))

                    # 学習開始
                    model.train_step.run(feed_dict={mail_doc2vec: np.array(
                        train_batch[0]), mail_class: np.array(train_batch[1])})

                elif len(lams) == 2 and first_flag == False:
                    first_flag = True
                elif len(lams) == 1:
                    train_batch, return_flag = random_batch(
                        trainset, 32, train_start_idx)

                    if return_flag == False:
                        train_start_idx += 1
                    elif return_flag == True:
                        train_start_idx = 0
                        train_ephoc += 1
                        print("train ephoc is {}".format(train_ephoc))

                    # 学習開始
                    model.train_step.run(feed_dict={mail_doc2vec: np.array(
                        train_batch[0]), mail_class: np.array(train_batch[1])})

            else:  # 学習回数が100回未満、かつ学習回数が奇数回の場合(つまりMVG)
                mvg_train_batch, return_flag = random_batch(
                    testsets[len(testsets)-1], 32, mvg_train_start_idx)

                if return_flag == False:
                    mvg_train_start_idx += 1
                elif return_flag == True:
                    mvg_train_start_idx = 0
                    mvg_train_ephoc += 1
                    print("mvg train ephoc is {}".format(mvg_train_ephoc))

                # 学習開始
                model.train_step.run(feed_dict={mail_doc2vec: np.array(
                    mvg_train_batch[0]), mail_class: np.array(mvg_train_batch[1])})

            # disp_freq(=20)回学習するごとに、テスト用メールに対する識別率を求める
            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l + 1)

                plots = []

                for task in range(len(testsets)):
                    test_batch = make_test_batch(testsets[task])

                    feed_dict = {mail_doc2vec: np.array(
                        test_batch[0]), mail_class: np.array(test_batch[1])}

                    # テスト用データに対する識別率を算出
                    test_accs[task][int(
                        iter / disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)

                    usage_guide = doc2vec_labels[task]

                    # 1回学習するごとに識別率を表示
                    if l == 0:
                        print("SGD" + " " + usage_guide + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))
                    else:
                        print("EWC" + " " + usage_guide + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))

                    plot_h, = plt.plot(range(
                        1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], marker='.', label=usage_guide)

                    plots.append(plot_h)

                plot_test_acc(plots)

                if l == 0:
                    plt.title("Stochastic Gradient Descent")
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

    plt.show()


def mvg(model, mode, mail_doc2vec, mail_class, doc2vec_dict, num_iter, disp_freq, sess, lams):
    train_start_idx = 0
    doc2vec_2005 = copy.copy(doc2vec_dict['2005'])
    doc2vec_2006 = copy.copy(doc2vec_dict['2006'])
    doc2vec_2007 = copy.copy(doc2vec_dict['2007'])

    # パラメータ読み込み
    model.restore(sess)

    # only SDG
    model.set_vanilla_loss()

    # 識別率を格納するためのリストを作成
    test_accs = [[], [], []]  # 順に2005,2006,2007(train_accsも同様)
    train_accs = [[], [], []]

    # どの年から学習するかの設定
    switch_flag = 2005

    for iteration in range(num_iter):
        if iteration < 700:  # # 2005年と2006年を20iterationごとに切り替えて学習
            if iteration == 0:  # 最初は2005年を学習
                train_doc2vec = copy.copy(
                    doc2vec_dict['2005'])
            elif iteration % 20 == 0 and switch_flag == 2005:  # 2006年の学習に切り替え
                train_doc2vec = copy.copy(
                    doc2vec_dict['2006'])
                switch_flag = 2006
            elif iteration % 20 == 0 and switch_flag == 2006:  # 2005年の学習に切り替え
                train_doc2vec = copy.copy(
                    doc2vec_dict['2005'])
                switch_flag = 2005
        elif iteration == 700:  # 2007年の学習に切り替え
            train_doc2vec = copy.copy(
                doc2vec_dict['2007'])

            if mode == 'EWC':
                # フィッシャー情報量の計算
                model.compute_fisher(
                    doc2vec_dict,
                    sess,
                    num_samples=400,
                    plot_diffs=False
                )

                model.stor()

                # 損失関数にEWCを適用
                model.update_ewc_loss(lams[1])

        train_batch, return_flag = random_batch(
            train_doc2vec, 32, train_start_idx)

        if return_flag == False:
            train_start_idx += 1
        elif return_flag == True:
            train_start_idx = 0

        # 学習開始
        model.train_step.run(feed_dict={mail_doc2vec: np.array(
            train_batch[0]), mail_class: np.array(train_batch[1])})

        # disp_freq(=20)回学習するごとに、テスト用メールに対する識別率を求める
        if iteration % disp_freq == 0:
            # 2005年のテスト用データに対する識別率を算出
            test_batch_2005 = make_test_batch(doc2vec_2005)

            feed_dict_test_2005 = {mail_doc2vec: np.array(
                test_batch_2005[0]), mail_class: np.array(test_batch_2005[1])}

            test_accs[0].append(
                model.accuracy.eval(feed_dict=feed_dict_test_2005))

            # 2005年の学習用データに対する識別率を算出
            train_batch_2005 = make_train_batch(doc2vec_2005)

            feed_dict_train_2005 = {mail_doc2vec: np.array(
                train_batch_2005[0]), mail_class: np.array(train_batch_2005[1])}

            train_accs[0].append(
                model.accuracy.eval(feed_dict=feed_dict_train_2005))

            # 2006年のテスト用データに対する識別率を算出
            test_batch_2006 = make_test_batch(doc2vec_2006)

            feed_dict_test_2006 = {mail_doc2vec: np.array(
                test_batch_2006[0]), mail_class: np.array(test_batch_2006[1])}

            test_accs[1].append(
                model.accuracy.eval(feed_dict=feed_dict_test_2006))

            # 2006年の学習用データに対する識別率を算出
            train_batch_2006 = make_train_batch(doc2vec_2006)

            feed_dict_train_2006 = {mail_doc2vec: np.array(
                train_batch_2006[0]), mail_class: np.array(train_batch_2006[1])}

            train_accs[1].append(
                model.accuracy.eval(feed_dict=feed_dict_train_2006))

            if iteration < 700:  # iterationが700未満の場合、2007年の分はNoneで埋める
                train_accs[2].append(None)
                test_accs[2].append(None)
            else:
                # 2007年のテスト用データに対する識別率を算出
                test_batch_2007 = make_test_batch(doc2vec_2007)

                feed_dict_test_2007 = {mail_doc2vec: np.array(
                    test_batch_2007[0]), mail_class: np.array(test_batch_2007[1])}

                test_accs[2].append(
                    model.accuracy.eval(feed_dict=feed_dict_test_2007))

                # 2007年の学習用データに対する識別率を算出
                train_batch_2007 = make_train_batch(doc2vec_2007)

                feed_dict_train_2007 = {mail_doc2vec: np.array(
                    train_batch_2007[0]), mail_class: np.array(train_batch_2007[1])}

                train_accs[2].append(
                    model.accuracy.eval(feed_dict=feed_dict_train_2007))

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
    plt.xticks(np.arange(0, 1201, 100))
    plt.ylim(bottom=0.40, top=1.01)
    if mode == 'NOTEWC':
        plt.title("MVG + SGD [test accuracy]")
    else:
        plt.title("MVG + EWC [test accuracy]")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

    # 学習用データの学習結果をプロット
    plt.plot(x_iter, train_accs[0], marker='.',
             label='2005')  # 2005年分の識別率をプロット
    plt.plot(x_iter, train_accs[1], marker='.',
             label='2006')  # 2006年分の識別率をプロット
    plt.plot(x_iter, train_accs[2], marker='.',
             label='2007')  # 2007年分の識別率をプロット

    plt.grid(which='major', color='black', linestyle='--')
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(np.arange(0, 1201, 100))
    plt.ylim(bottom=0.40, top=1.01)
    if mode == 'NOTEWC':
        plt.title("MVG + SGD [train accuracy]")
    else:
        plt.title("MVG + EWC [train accuracy]")
    plt.xlabel("Iterations")
    plt.ylabel("Train Accuracy")
    plt.show()

    print("=" * 10 + "test accs" + "=" * 10)
    print("2005 test accs", test_accs[0][-1])
    print("2006 test accs", test_accs[1][-1])
    print("2007 test accs", test_accs[2][-1])
    print("=" * 10 + "test accs" + "=" * 10)
    print("2005 train accs", train_accs[0][-1])
    print("2006 train accs", train_accs[1][-1])
    print("2007 train accs", train_accs[2][-1])


def main():
    """
    read_doc2vec.pyにより実行される関数
    """
    common_path = '/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_year_doc2vec_model/'
    model_info_orderdict = OrderedDict({
        '2005': common_path + '2005_2000_mails_doc2vec.csv',
        '2006': common_path + '2006_2000_mails_doc2vec.csv',
        '2007': common_path + '2007_2000_mails_doc2vec.csv'
    })

    model_doc2vec_dict = dict()
    model_doc2vec_learned_list = []
    doc2vec_labels_list = []

    PARAMETERS = parameter()

    for doc2vec_label, file_path in model_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            model_doc2vec_dict[doc2vec_label] = [
                [float(v) for v in row] for row in reader]

    sess = tf.compat.v1.InteractiveSession()

    mail_doc2vec = tf.compat.v1.placeholder(
        tf.float32, shape=[None, 300])  # doc2vecの場合300
    mail_class = tf.compat.v1.placeholder(
        tf.float32, shape=[None, 2])  # 教師ラベルの種類　ham:[1,0],spam:[0,1]

    # ニューラルネットワークモデルの構築(モデルの具体的な構成はmodel_for_doc2vec.pyを参照)
    model = Model(mail_doc2vec, mail_class)

    sess.run(tf.compat.v1.global_variables_initializer())

    PARAMETERS.set_lams(50)
    mvg(
        model,
        'EWC',
        mail_doc2vec,
        mail_class,
        model_doc2vec_dict,
        PARAMETERS.ITERATIONS,
        PARAMETERS.DISP_FREQ,
        sess,
        PARAMETERS.lams
    )


if __name__ == "__main__":
    main()
