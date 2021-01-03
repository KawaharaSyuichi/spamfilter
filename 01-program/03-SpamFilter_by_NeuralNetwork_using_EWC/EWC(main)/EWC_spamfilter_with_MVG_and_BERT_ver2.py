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
from model_for_bert_ver2 import Model

random.seed(2020)
tf.compat.v1.set_random_seed(2020)
warnings.filterwarnings(
    'ignore', category=matplotlib.MatplotlibDeprecationWarning)


class parameter:
    """
    ニューラルネットワークで使用するパラメータ
    """

    def __init__(self):
        self.ITERATIONS = 1320  # 元は520
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
    plt.yticks(np.arange(0.45, 1.001, 0.01))
    plt.xticks(np.arange(0, 1320, 100))  # 元は0,520,20
    plt.ylim(0.89, 1.001)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def train_task(model, num_iter, disp_freq, trainset, doc2vec_labels, testsets, mail_doc2vec, mail_class, sess, lams):
    """
    学習と識別の実行
    """
    train_start_idx = 0
    train_ephoc = 0

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


def main():
    common_path = '/Users/kawahara/Documents/01-programming/00-大学研究/02-result/BERT_vector/'
    model_info_orderdict = OrderedDict({
        '2005': common_path + 'BERT_2005_body_512.csv',
        '2006': common_path + 'BERT_2006_body_512.csv',
        '2007': common_path + 'BERT_2007_body_512.csv'
    })

    model_header_info_orderdict = OrderedDict({
        '2005': common_path + 'BERT_2005_header_512.csv',
        '2006': common_path + 'BERT_2006_header_512.csv',
        '2007': common_path + 'BERT_2007_header_512.csv'
    })

    model_bert_dict = dict()
    model_body_bert_dict = dict()
    model_header_bert_dict = dict()
    model_doc2vec_learned_list = []
    doc2vec_labels_list = []

    PARAMETERS = parameter()

    for doc2vec_label, file_path in model_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            model_body_bert_dict[doc2vec_label] = [
                [float(v) for v in row] for row in reader]

    for doc2vec_label, file_path in model_header_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            model_header_bert_dict[doc2vec_label] = [
                [float(v) for v in row] for row in reader]

    for year in model_info_orderdict.keys():
        header_bert_list = model_header_bert_dict[year]
        body_bert_list = model_body_bert_dict[year]
        bert = []
        for header_bert, body_bert in zip(header_bert_list, body_bert_list):
            header_bert.extend(body_bert)
            bert.append(header_bert)

        model_bert_dict[year] = bert

    sess = tf.compat.v1.InteractiveSession()

    mail_doc2vec = tf.compat.v1.placeholder(
        tf.float32, shape=[None, 1536])  # doc2vecの場合768
    mail_class = tf.compat.v1.placeholder(
        tf.float32, shape=[None, 2])  # 教師ラベルの種類　ham:[1,0],spam:[0,1]

    # ニューラルネットワークモデルの構築(モデルの具体的な構成はmodel_for_doc2vec.pyを参照)
    model = Model(mail_doc2vec, mail_class)

    sess.run(tf.compat.v1.global_variables_initializer())

    for doc2vec_label in model_info_orderdict.keys():
        doc2vec_labels_list.append(doc2vec_label)
        model_doc2vec_learned_list.append(model_bert_dict[doc2vec_label])

        train_task(model, PARAMETERS.ITERATIONS, PARAMETERS.DISP_FREQ,
                   model_bert_dict[doc2vec_label], doc2vec_labels_list, model_doc2vec_learned_list, mail_doc2vec, mail_class, sess, PARAMETERS.lams)

        # 各ラベルのフィッシャー情報量の計算
        model.compute_fisher(model_bert_dict[doc2vec_label], sess,
                             num_samples=200, plot_diffs=False)

        # 重みとバイアスのパラメータ保存
        model.stor()

        if len(PARAMETERS.lams) == 1:
            PARAMETERS.set_lams(2000)  # 50


if __name__ == "__main__":
    main()