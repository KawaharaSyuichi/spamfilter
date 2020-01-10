# coding: UTF-8
import gensim
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
from model_for_hyperparameter import Model
from gensim.models import Doc2Vec


train_ephoc = 0
test_ephoc = 0


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
        if month_flag == 0:
            # english(2000個のデータ)(spam:0~999,ham:1000~1999)
            # March(4000個のデータ)(spam:0~1999,ham:2000~3999)

            spam_half_start = 100 + start_idx * half_batch_size
            spam_half_end = 100 + (start_idx + 1) * half_batch_size
            ham_half_start = 2100 + start_idx * half_batch_size
            ham_half_end = 2100 + (start_idx + 1) * half_batch_size

            if spam_half_end > 1000:
                idx = [num for num in range(spam_half_start, 1000)]
                idx.extend(random.sample(range(100, spam_half_start),
                                         k=half_batch_size - len(idx)))

                spam_half = trainset[100:1000]
                ham_half = trainset[2100:3000]
                random.shuffle(spam_half)
                random.shuffle(ham_half)
                trainset[100:1000] = spam_half
                trainset[2100:3000] = ham_half

                idx_1 = [num for num in range(ham_half_start, 3000)]
                idx_1.extend(random.sample(range(2100, ham_half_start),
                                           k=half_batch_size - len(idx_1)))

                return_flag = True
            else:
                idx = [num for num in range(spam_half_start, spam_half_end)]
                random.shuffle(idx)
                idx_1 = [num for num in range(ham_half_start, ham_half_end)]
                random.shuffle(idx_1)

            idx.extend(idx_1)
        elif month_flag == 1:
            # April(4000呼のデータ)(spam:0~1999,ham:2000~39999)
            # idx = random.sample(range(500), k=int(batch_size / 2))
            # idx_1 = random.sample(range(1000, 1500), k=int(batch_size / 2))

            spam_half_start = 100 + start_idx * half_batch_size
            spam_half_end = 100 + (start_idx + 1) * half_batch_size
            ham_half_start = 2100 + start_idx * half_batch_size
            ham_half_end = 2100 + (start_idx + 1) * half_batch_size

            if spam_half_end > 1000:
                idx = [num for num in range(spam_half_start, 1000)]
                idx.extend(random.sample(range(100, spam_half_start),
                                         k=half_batch_size - len(idx)))

                spam_half = trainset[100:1000]
                ham_half = trainset[2100:3000]
                random.shuffle(spam_half)
                random.shuffle(ham_half)
                trainset[100:1000] = spam_half
                trainset[2100:3000] = ham_half

                idx_1 = [num for num in range(ham_half_start, 3000)]
                idx_1.extend(random.sample(range(2100, ham_half_start),
                                           k=half_batch_size - len(idx_1)))

                return_flag = True
            else:
                idx = [num for num in range(spam_half_start, spam_half_end)]
                random.shuffle(idx)
                idx_1 = [num for num in range(ham_half_start, ham_half_end)]
                random.shuffle(idx_1)

            idx.extend(idx_1)
        else:
            # May(4000呼のデータ)(spam:0~1999,ham:2000~3999)
            # idx = random.sample(range(500), k=int(batch_size / 2))
            # idx_1 = random.sample(range(1000, 1500), k=int(batch_size / 2))
            spam_half_start = 100 + start_idx * half_batch_size
            spam_half_end = 100 + (start_idx + 1) * half_batch_size
            ham_half_start = 2100 + start_idx * half_batch_size
            ham_half_end = 2100 + (start_idx + 1) * half_batch_size

            if spam_half_end > 1000:
                idx = [num for num in range(spam_half_start, 1000)]
                idx.extend(random.sample(range(100, spam_half_start),
                                         k=half_batch_size - len(idx)))

                spam_half = trainset[100:1000]
                ham_half = trainset[2100:3000]
                random.shuffle(spam_half)
                random.shuffle(ham_half)
                trainset[100:1000] = spam_half
                trainset[2100:3000] = ham_half

                idx_1 = [num for num in range(ham_half_start, 3000)]
                idx_1.extend(random.sample(range(2100, ham_half_start),
                                           k=half_batch_size - len(idx_1)))

                return_flag = True
            else:
                idx = [num for num in range(spam_half_start, spam_half_end)]
                random.shuffle(idx)
                idx_1 = [num for num in range(ham_half_start, ham_half_end)]
                random.shuffle(idx_1)

            idx.extend(idx_1)
    elif train_flag == False:  # 検証用
        if month_flag == 0:  # March
            idx = [num for num in range(1000, 2000)]
            idx_1 = [num for num in range(3000, 4000)]

            idx.extend(idx_1)

        elif month_flag == 1:  # April

            idx = [num for num in range(1000, 2000)]
            idx_1 = [num for num in range(3000, 4000)]

            idx.extend(idx_1)
        else:  # May
            idx = [num for num in range(1000, 2000)]
            idx_1 = [num for num in range(3000, 3700)]

            idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num <= 1899:
            docflag.append([0.0, 1.0])  # spam
        else:
            docflag.append([1.0, 0.0])  # ham

    batch.append(docvec)
    batch.append(docflag)

    if return_flag == True and train_flag == True:
        train_ephoc += 1
        if train_ephoc % 100 == 0:
            print("#" * 50)
            print("train_ephoc:{}".format(train_ephoc))
            print("#" * 50)
    elif return_flag == True and train_flag == False:
        test_ephoc += 1
        print("*" * 50)
        print("test_ephoc:{}".format(test_ephoc))
        print("*" * 50)

    return batch, return_flag


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(np.arange(0, 310, 50))
    plt.ylim(0.0, 1.01)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def plot_validation_acc(validation_acc, unit, learning_rate_list):
    label = list()
    for learning_rate in learning_rate_list:
        label.append(str(learning_rate))

    print(label)

    plt.xlabel("(unit , learnign rate)")
    plt.ylabel("Accuracy")
    plt.ylim(0.945, 0.951)
    plt.xticks([0, 1, 2, 3], label)
    plt.plot(validation_acc)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.show()


def train_task(validation_acc, learning_rate, model, num_iter, disp_freq, trainset, testsets, mail_doc2vec, mail_class, month_flag, lams=[0]):
    train_start_idx = 0

    March_start_idx = 0
    April_start_idx = 0
    May_start_idx = 0

    for l in range(len(lams)):
        first_flag = False
        # lams[l] sets weight on old task(s)
        # reassign optimal weights from previous training session
        model.restore(sess)

        if(lams[l] == 0):
            model.set_vanilla_loss(learning_rate)
        else:
            model.update_ewc_loss(lams[l])

        # initialize test accuracy array for each task
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter / disp_freq)))

        # train on current task
        for iter in range(num_iter):
            # (trainset,batch_size,laguage_flag):manth_flag={0:March,1:April,2:May}

            if len(lams) == 2 and first_flag == True:  # 初の学習を行う場合
                train_batch, return_flag = random_batch(
                    trainset, 80, month_flag, train_start_idx, train_flag=True)

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
                    trainset, 80, month_flag, train_start_idx, train_flag=True)

                if return_flag == False:
                    train_start_idx += 1
                elif return_flag == True:
                    train_start_idx = 0

                model.train_step.run(feed_dict={mail_doc2vec: np.array(
                    train_batch[0]), mail_class: np.array(train_batch[1])})

            if iter % disp_freq == 0:  # (disp_freq回学習するごとに、識別率を求める)
                plt.subplot(1, len(lams), l + 1)

                #plots = []
                #colors = ['r', 'b', 'c']

                for task in range(len(testsets)):
                    # task==0:3月
                    # task==1:4月
                    # task==2:5月
                    # テストデータの推定値
                    if task == 0:  # 3月
                        test_batch, return_flag = random_batch(
                            testsets[task], 2000, 0, March_start_idx, train_flag=False)

                        if return_flag == False:
                            March_start_idx += 1
                        else:
                            March_start_idx = 0

                    elif task == 1:  # 4月
                        test_batch, return_flag = random_batch(
                            testsets[task], 2000, 1, April_start_idx, train_flag=False)

                        if return_flag == False:
                            April_start_idx += 1
                        else:
                            April_start_idx = 0

                    elif task == 2:  # 5月
                        test_batch, return_flag = random_batch(
                            testsets[task], 2000, 2, May_start_idx, train_flag=False)

                    feed_dict = {mail_doc2vec: np.array(
                        test_batch[0]), mail_class: np.array(test_batch[1])}

                    test_accs[task][int(
                        iter / disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)

                    if task == 0:  # March task
                        c = "March"
                    elif task == 1:  # April task
                        c = "April"
                    elif task == 2:  # May task
                        c = "May"

                    if l == 0:
                        print("SGD" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))
                    else:
                        print("EWC" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))

                    # plot_h, = plt.plot(range(1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], colors[task], label=c)

                    # plots.append(plot_h)

                # plot_test_acc(plots)

                """
                if l == 0:
                    plt.title("Stochastic Gradient Descent")  # 確率的勾配降下法
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)
                """

    validation_acc.append(test_accs[0][-1])

    # plt.show()


if __name__ == "__main__":
    unit_num_list = [50, 100, 150, 200, 250, 300]
    learning_rate_list = [0.01, 0.1, 1, 10]
    inner_layer_list = list(range(1, 11))

    label = list()
    label_dict = dict()

    for unit_num in unit_num_list:
        for learning_rate in learning_rate_list:
            label.append("(" + str(unit_num) + "," + str(learning_rate) + ")")
        # ↓単純にlabelとしてしまうと次の行の動作によって，辞書内のlistまで消えてしまう．
        label_dict[unit_num] = label[:]
        label.clear()

    validation_acc = []

    model_path = "/Users/kawahara/Documents/01-programming/00-大学研究/01-MailSearch/doc2vec_models/"

    # May_spam_model_list:12130
    # May_ham_model_list:5700
    May_spam_model_doc2vec = Doc2Vec.load(model_path + "doc2vec_of_May.model")
    May_spam_model_list = [May_spam_model_doc2vec.docvecs[num]
                           for num in range(len(May_spam_model_doc2vec.docvecs))]

    May_ham_model_doc2vec = Doc2Vec.load(
        model_path + "ham_doc2vec_of_May.model")
    May_ham_model_list = [May_ham_model_doc2vec.docvecs[num]
                          for num in range(len(May_ham_model_doc2vec.docvecs))]

    March_mail_list = May_spam_model_list[0:2000]
    March_mail_list.extend(May_ham_model_list[0:2000])

    April_mail_list = May_spam_model_list[2000:4000]
    April_mail_list.extend(May_ham_model_list[2000:4000])

    May_mail_list = May_spam_model_list[4000:6000]
    May_mail_list.extend(May_ham_model_list[4000:5700])

    for unit in unit_num_list:
        for learning_rate in learning_rate_list:
            sess = tf.InteractiveSession()

            mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
            mail_class = tf.placeholder(
                tf.float32, shape=[None, 2])  # ham:[1,0],spam:[0,1]

            # simple 2-layer network
            model = Model(unit, learning_rate, mail_doc2vec, mail_class)

            sess.run(tf.global_variables_initializer())

            # training 1st task
            # 最期から二個目の引き数:言語の選択={0:english,1:japanese}
            train_task(validation_acc, learning_rate, model, 320, 20, March_mail_list, [
                       March_mail_list], mail_doc2vec, mail_class, 0, lams=[0])
            print(unit, learning_rate)

        plot_validation_acc(validation_acc, unit, label_dict[unit])
        validation_acc.clear()

    print(validation_acc)


"""
# Fisher information
# use validation set for Fisher computation
model.compute_fisher(March_mail_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

# save current optimal weights
model.star()

test_ephoc = 0
train_ephoc = 0

print("March task finished")

# training 2nd task
train_task(model, 320, 20, April_mail_list, [
           March_mail_list, April_mail_list], mail_doc2vec, mail_class, 1, lams=[0, 40])

# Fisher information
# use validation set for Fisher computation
model.compute_fisher(April_mail_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

# save current optimal weights
model.star()

test_ephoc = 0
train_ephoc = 0

print("April task finished")

# training 3nd task
train_task(model, 320, 20, May_mail_list, [
           March_mail_list, April_mail_list, May_mail_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

print("May task finished")
"""
