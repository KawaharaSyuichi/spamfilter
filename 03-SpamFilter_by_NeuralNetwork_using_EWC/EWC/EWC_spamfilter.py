# coding: UTF-8

import gensim
import random
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
#from model_for_doc2vec_deep import Model
from model_for_doc2vec import Model
from gensim.models import Doc2Vec

train_ephoc = 0


def random_batch(trainset, batch_size, month_flag, start_idx, train_flag=True):
    half_batch_size = int(batch_size / 2)
    return_flag = False
    global train_ephoc
    global test_ephoc

    batch = []
    docvec = []
    docflag = []
    idx = []

    if train_flag == True:  # 学習用
        # spam[0:5000]
        # ham[5000:10000]
        spam_half_start = 100 + start_idx * half_batch_size
        spam_half_end = 100 + (start_idx + 1) * half_batch_size
        ham_half_start = 5100 + start_idx * half_batch_size
        ham_half_end = 5100 + (start_idx + 1) * half_batch_size
        if spam_half_end > 2000:
            idx = [num for num in range(spam_half_start, 2000)]
            idx.extend(random.sample(range(100, spam_half_start),
                                     k=half_batch_size - len(idx)))
            spam_half = trainset[100:2000]
            ham_half = trainset[5100:7000]
            random.shuffle(spam_half)
            random.shuffle(ham_half)
            trainset[100:2000] = spam_half
            trainset[5100:7000] = ham_half
            idx_1 = [num for num in range(ham_half_start, 7000)]
            idx_1.extend(random.sample(range(5100, ham_half_start),
                                       k=half_batch_size - len(idx_1)))
            return_flag = True
        else:
            idx = [num for num in range(spam_half_start, spam_half_end)]
            random.shuffle(idx)
            idx_1 = [num for num in range(ham_half_start, ham_half_end)]
            random.shuffle(idx_1)
        idx.extend(idx_1)

    elif train_flag == False:  # 検証用
        idx = [num for num in range(2000, 4000)]
        idx_1 = [num for num in range(7000, 9000)]
        idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

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


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="lower right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(np.arange(0, 1020, 50))
    plt.ylim(0.0, 1.01)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def train_task(model, num_iter, disp_freq, trainset, testsets, mail_doc2vec, mail_class, month_flag, lams=[0]):
    train_start_idx = 0

    start_idx_2005 = 0
    start_idx_2006 = 0
    start_idx_2007 = 0

    for l in range(len(lams)):
        first_flag = False
        # lams[l] sets weight on old task(s)
        # reassign optimal weights from previous training session
        model.restore(sess)

        if(lams[l] == 0):
            model.set_vanilla_loss()
        else:
            model.update_ewc_loss(lams[l])

        # initialize test accuracy array for each task
        test_accs = []
        last_test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter / disp_freq)))
            last_test_accs.append(np.zeros(int(num_iter / disp_freq)))

        # train on current task
        for iter in range(num_iter):
            # (trainset,batch_size,laguage_flag):manth_flag={0:March,1:April,2:May}

            if len(lams) == 2 and first_flag == True:  # 初の学習を行う場合
                train_batch, return_flag = random_batch(
                    trainset, 100, month_flag, train_start_idx, train_flag=True)

                if return_flag == False:
                    train_start_idx += 1
                elif return_flag == True:
                    train_start_idx = 0

                model.train_step.run(feed_dict={mail_doc2vec: np.array(
                    train_batch[0]), mail_class: np.array(train_batch[1])})

            elif len(lams) == 2 and first_flag == False:
                first_flag = True
            elif len(lams) == 1:
                train_batch, return_flag = random_batch(
                    trainset, 100, month_flag, train_start_idx, train_flag=True)

                if return_flag == False:
                    train_start_idx += 1
                elif return_flag == True:
                    train_start_idx = 0

                model.train_step.run(feed_dict={mail_doc2vec: np.array(
                    train_batch[0]), mail_class: np.array(train_batch[1])})

            if iter % disp_freq == 0:  # (disp_freq回学習するごとに、識別率を求める)
                plt.subplot(1, len(lams), l + 1)

                plots = []
                colors = ['r', 'b', 'c']

                for task in range(len(testsets)):
                    # テストデータの推定値
                    if task == 0:  # 2005年
                        test_batch, return_flag = random_batch(
                            testsets[task], 4000, 0, start_idx_2005, train_flag=False)

                        if return_flag == False:
                            start_idx_2005 += 1
                        else:
                            start_idx_2005 = 0

                    elif task == 1:  # 2006年
                        test_batch, return_flag = random_batch(
                            testsets[task], 4000, 1, start_idx_2006, train_flag=False)

                        if return_flag == False:
                            start_idx_2006 += 1
                        else:
                            start_idx_2006 = 0

                    elif task == 2:  # 2007年
                        test_batch, return_flag = random_batch(
                            testsets[task], 4000, 2, start_idx_2006, train_flag=False)

                        if return_flag == False:
                            start_idx_2007 += 1
                        else:
                            start_idx_2007 = 0

                    feed_dict = {mail_doc2vec: np.array(
                        test_batch[0]), mail_class: np.array(test_batch[1])}

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
                    plt.title("Stochastic Gradient Descent")  # 確率的勾配降下法
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

    plt.show()


model_path = "/Users/kawahara/Documents/01-programming/00-大学研究/01-MailSearch/trec_doc2vec/trec_year_doc2vec_model/"

# May_spam_model_list:12130
# May_ham_model_list:5700
doc2vec_2005 = Doc2Vec.load(model_path + "2005_doc2vec.model")
mail_2005_list = [doc2vec_2005.docvecs[num]
                  for num in range(len(doc2vec_2005.docvecs))]

doc2vec_2006 = Doc2Vec.load(model_path + "2006_doc2vec.model")
mail_2006_list = [doc2vec_2006.docvecs[num]
                  for num in range(len(doc2vec_2006.docvecs))]

doc2vec_2007 = Doc2Vec.load(model_path + "2007_doc2vec.model")
mail_2007_list = [doc2vec_2007.docvecs[num]
                  for num in range(len(doc2vec_2007.docvecs))]

print(len(mail_2005_list), len(mail_2006_list), len(mail_2007_list))

sess = tf.InteractiveSession()

mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
mail_class = tf.placeholder(
    tf.float32, shape=[None, 2])  # ham:[1,0],spam:[0,1]

model = Model(mail_doc2vec, mail_class)  # simple 2-layer network

sess.run(tf.global_variables_initializer())

# training 1st task
# 最期から二個目の引き数:言語の選択={0:english,1:japanese}
train_task(model, 1020, 20, mail_2005_list, [
           mail_2005_list], mail_doc2vec, mail_class, 0, lams=[0])
print('train_task finished')

# Fisher information
# use validation set for Fisher computation
model.compute_fisher(mail_2005_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

# save current optimal weights
model.star()

train_ephoc = 0

print("2005 task finished")

# training 2nd task
train_task(model, 1020, 20, mail_2006_list, [
           mail_2005_list, mail_2006_list], mail_doc2vec, mail_class, 1, lams=[0, 100])

# Fisher information
# use validation set for Fisher computation
model.compute_fisher(mail_2006_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")


# save current optimal weights
model.star()

train_ephoc = 0

print("2006 task finished")

# training 3nd task
train_task(model, 1020, 20, mail_2007_list, [
           mail_2005_list, mail_2006_list, mail_2007_list], mail_doc2vec, mail_class, 2, lams=[0, 100])

print("2007 task finished")
