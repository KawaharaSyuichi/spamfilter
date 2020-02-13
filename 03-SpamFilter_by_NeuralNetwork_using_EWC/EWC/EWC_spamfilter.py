# tensorflow version: 1.14.0
import sys
import gensim
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from gensim.models import Doc2Vec
from model_for_doc2vec import Model

train_ephoc = 0


def random_batch(trainset, batch_size, start_idx):
    """
    学習用のバッチ作成
    """
    half_batch_size = int(batch_size / 2)
    return_flag = False
    global train_ephoc

    batch = []
    docvec = []
    docflag = []
    idx = []

    spam_half_start = 100 + start_idx * half_batch_size
    spam_half_end = 100 + (start_idx + 1) * half_batch_size
    ham_half_start = 5100 + start_idx * half_batch_size
    ham_half_end = 5100 + (start_idx + 1) * half_batch_size
    if spam_half_end > 2500:  # 学習用データセットを一通り学習した場合(エポック数が1増える)
        idx = [num for num in range(spam_half_start, 2500)]
        idx.extend(random.sample(range(100, spam_half_start),
                                 k=half_batch_size - len(idx)))
        idx_1 = [num for num in range(ham_half_start, 7500)]
        idx_1.extend(random.sample(range(5100, ham_half_start),
                                   k=half_batch_size - len(idx_1)))

        # 学習用データセットをシャッフル
        spam_half = trainset[100:2500]
        ham_half = trainset[5100:7500]
        random.shuffle(spam_half)
        random.shuffle(ham_half)
        trainset[100:2500] = spam_half
        trainset[5100:7500] = ham_half

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
        if num < 5000:
            docflag.append([0.0, 1.0])  # spam
        else:
            docflag.append([1.0, 0.0])  # ham

    batch.append(docvec)
    batch.append(docflag)

    if return_flag == True:
        train_ephoc += 1
        print("train_ephoc:{}".format(train_ephoc))

    return batch, return_flag


def make_test_batch(trainset):
    """
    テスト用のバッチ作成
    """
    batch = []
    docvec = []
    docflag = []

    # テスト用バッチでは学習用で用いなかった残りのデータを全て使用する
    idx = [num for num in range(2500, 5000)]  # スパムメール分
    idx_1 = [num for num in range(7500, 10000)]  # 正規メール分
    idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num < 5000:
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
    plt.xticks(np.arange(0, 1020, 50))
    plt.ylim(0.0, 1.01)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def train_task(model, num_iter, disp_freq, trainset, testsets, mail_doc2vec, mail_class, lams=[0]):
    train_start_idx = 0

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

                # 学習開始
                model.train_step.run(feed_dict={mail_doc2vec: np.array(
                    train_batch[0]), mail_class: np.array(train_batch[1])})

            # disp_freq(=20)回学習するごとに、テスト用メールに対する識別率を求める
            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l + 1)

                plots = []
                colors = ['r', 'b', 'c']

                for task in range(len(testsets)):
                    test_batch = make_test_batch(testsets[task])

                    feed_dict = {mail_doc2vec: np.array(
                        test_batch[0]), mail_class: np.array(test_batch[1])}

                    # テスト用データに対する識別率を算出
                    test_accs[task][int(
                        iter / disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)

                    if task == 0:  # 2005 task
                        c = "2005"
                    elif task == 1:  # 2006 task
                        c = "2006"
                    elif task == 2:  # 2007 task
                        c = "2007"

                    if l == 0:
                        print("last SGD" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))
                    else:
                        print("last EWC" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))

                    plot_h, = plt.plot(range(
                        1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], colors[task], label=c)

                    plots.append(plot_h)

                plot_test_acc(plots)

                if l == 0:
                    plt.title("Stochastic Gradient Descent")
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

    plt.show()


model_path = "/Users/kawahara/Documents/01-programming/00-大学研究/01-MailSearch/trec_doc2vec/trec_year_doc2vec_model/"

doc2vec_2005 = Doc2Vec.load(model_path + "2005_doc2vec.model")
mail_2005_list = [doc2vec_2005.docvecs[num]
                  for num in range(len(doc2vec_2005.docvecs))]

doc2vec_2006 = Doc2Vec.load(model_path + "2006_doc2vec.model")
mail_2006_list = [doc2vec_2006.docvecs[num]
                  for num in range(len(doc2vec_2006.docvecs))]

doc2vec_2007 = Doc2Vec.load(model_path + "2007_doc2vec.model")
mail_2007_list = [doc2vec_2007.docvecs[num]
                  for num in range(len(doc2vec_2007.docvecs))]

sess = tf.InteractiveSession()

mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
mail_class = tf.placeholder(
    tf.float32, shape=[None, 2])  # ラベルの種類　ham:[1,0],spam:[0,1]

# ニューラルネットワークモデルの構築(モデルの具体的な構成はmodel_for_doc2vec.pyを参照)
model = Model(mail_doc2vec, mail_class)

sess.run(tf.global_variables_initializer())

# 2005年メールデータセットの学習
train_task(model, 1020, 20, mail_2005_list, [
           mail_2005_list], mail_doc2vec, mail_class, lams=[0])
print('2005 train finished')

# 2005年分のフィッシャー情報量の計算
model.compute_fisher(mail_2005_list, sess,
                     num_samples=200, plot_diffs=True)
print("2005 model.compute_fisher finished")

# 重みとバイアスのパラメータ保存
model.stor()

# エポック数のリセット
train_ephoc = 0

print("2005 task finished")

# 2006年メールデータセットの追加学習
# 最後の引数lams=[0,100]のうち、二つ目の値(100)はEWCを用いるときに使用するパラメータλ(詳しい説明はREADME.mdを参照)
train_task(model, 1020, 20, mail_2006_list, [
           mail_2005_list, mail_2006_list], mail_doc2vec, mail_class, lams=[0, 100])

# 2006年分のフィッシャー情報量の計算
model.compute_fisher(mail_2006_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.stor()

train_ephoc = 0

print("2006 task finished")

# 2007年メールデータセットの追加学習
# 追加学習する最後のデータセットのため、フィッシャー情報量の計算はしない
train_task(model, 1020, 20, mail_2007_list, [
           mail_2005_list, mail_2006_list, mail_2007_list], mail_doc2vec, mail_class, lams=[0, 100])

print("2007 task finished")
