import sys
import csv
import gensim
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from gensim.models import Doc2Vec
from model_for_doc2vec import Model


class parameter:
    """
    ニューラルネットワークで使用するパラメータ
    """

    def __init__(self):
        self.ITERATIONS = 520
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
    ham_half_start = 2100 + start_idx * half_batch_size
    ham_half_end = 2100 + (start_idx + 1) * half_batch_size

    if spam_half_end > 1000:  # 学習用データセットを一通り学習した場合(エポック数が1増える)
        idx = [num for num in range(spam_half_start, 1000)]
        idx.extend(random.sample(range(100, spam_half_start),
                                 k=half_batch_size - len(idx)))
        idx_1 = [num for num in range(ham_half_start, 3000)]
        idx_1.extend(random.sample(range(2100, ham_half_start),
                                   k=half_batch_size - len(idx_1)))

        # 学習用データセットをシャッフル
        spam_half = trainset[100:1000]
        ham_half = trainset[2100:3000]
        random.shuffle(spam_half)
        random.shuffle(ham_half)
        trainset[100:1000] = spam_half
        trainset[2100:3000] = ham_half

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
        if num < 2000:
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
    idx = [num for num in range(1000, 2000)]  # スパムメール分
    idx_1 = [num for num in range(3000, 4000)]  # 正規メール分
    idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num < 2000:
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
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(np.arange(0, 520, 50))
    plt.ylim(0.0, 1.01)
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
                    trainset, 100, train_start_idx)

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
                    trainset, 100, train_start_idx)

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
                        1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], label=usage_guide)

                    plots.append(plot_h)

                plot_test_acc(plots)

                if l == 0:
                    plt.title("Stochastic Gradient Descent")
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

    plt.show()


def main(model_info_orderdict):
    """
    read_doc2vec.pyにより実行される関数
    """

    model_doc2vec_dict = dict()
    model_doc2vec_learned_list = []
    doc2vec_labels_list = []

    PARAMETERS = parameter()

    for doc2vec_label, file_path in model_info_orderdict.items():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            model_doc2vec_dict[doc2vec_label] = [row for row in reader]

    sess = tf.InteractiveSession()

    mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
    mail_class = tf.placeholder(
        tf.float32, shape=[None, 2])  # 教師ラベルの種類　ham:[1,0],spam:[0,1]

    # ニューラルネットワークモデルの構築(モデルの具体的な構成はmodel_for_doc2vec.pyを参照)
    model = Model(mail_doc2vec, mail_class)

    sess.run(tf.global_variables_initializer())

    for doc2vec_label in model_info_orderdict.keys():
        doc2vec_labels_list.append(doc2vec_label)
        model_doc2vec_learned_list.append(model_doc2vec_dict[doc2vec_label])

        train_task(model, PARAMETERS.ITERATIONS, PARAMETERS.DISP_FREQ,
                   model_doc2vec_dict[doc2vec_label], doc2vec_labels_list, model_doc2vec_learned_list, mail_doc2vec, mail_class, sess, PARAMETERS.lams)

        # 各ラベルのフィッシャー情報量の計算
        model.compute_fisher(model_doc2vec_dict[doc2vec_label], sess,
                             num_samples=200, plot_diffs=True)

        # 重みとバイアスのパラメータ保存
        model.stor()

        if len(PARAMETERS.lams) == 1:
            PARAMETERS.set_lams(100)
