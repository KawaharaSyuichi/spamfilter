# coding: UTF-8
import gensim
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
from model_for_doc2vec import Model
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
        # 2000個のデータ(spam:0~999,ham:1000~1999)
        spam_half_start = 100 + start_idx * half_batch_size
        spam_half_end = 100 + (start_idx + 1) * half_batch_size
        ham_half_start = 1100 + start_idx * half_batch_size
        ham_half_end = 1100 + (start_idx + 1) * half_batch_size

        if spam_half_end > 1000:
            idx = [num for num in range(spam_half_start, 1000)]
            idx.extend(random.sample(range(100, spam_half_start),
                                     k=half_batch_size - len(idx)))
            idx_1 = [num for num in range(ham_half_start, 2000)]
            idx_1.extend(random.sample(range(1100, ham_half_start),
                                       k=half_batch_size - len(idx_1)))

            spam_half = trainset[100:1000]
            ham_half = trainset[1100:2000]
            random.shuffle(spam_half)
            random.shuffle(ham_half)
            trainset[100:1000] = spam_half
            trainset[1100:2000] = ham_half

            return_flag = True
        else:
            idx = [num for num in range(spam_half_start, spam_half_end)]
            random.shuffle(idx)
            idx_1 = [num for num in range(ham_half_start, ham_half_end)]
            random.shuffle(idx_1)

        idx.extend(idx_1)

    elif train_flag == False:  # 検証用
        idx = [num for num in range(100, 1000)]
        idx_1 = [num for num in range(1100, 2000)]

        idx.extend(idx_1)

    for num in idx:
        docvec.append(trainset[num])

        if num < 1000:
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

    return batch, return_flag


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="lower right")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(np.arange(0, 310, 50))
    plt.ylim(0.0, 1.01)
    plt.grid(which='major', color='black', linestyle='-')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def train_task(model, num_iter, disp_freq, trainset, testsets, mail_doc2vec, mail_class, month_flag, lams=[0]):
    train_start_idx = 0

    March_start_idx = 0

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

                plots = []
                #colors = ['r', 'b', 'c']

                for task in range(len(testsets)):
                    # テストデータの推定値
                    test_batch, return_flag = random_batch(
                        testsets[task], 2000, 0, March_start_idx, train_flag=False)

                    feed_dict = {mail_doc2vec: np.array(
                        test_batch[0]), mail_class: np.array(test_batch[1])}

                    test_accs[task][int(
                        iter / disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)

                    if task == 0:
                        c = "english_1"
                    elif task == 1:
                        c = "japanese_1"
                    elif task == 2:
                        c = "english_2"
                    elif task == 3:
                        c = "japanese_2"
                    elif task == 4:
                        c = "english_3"
                    elif task == 5:
                        c = "japanese_3"
                    elif task == 6:
                        c = "english_4"
                    elif task == 7:
                        c = "japanese_4"
                    elif task == 8:
                        c = "english_5"
                    elif task == 9:
                        c = "japanese_5"
                    elif task == 10:
                        c = "english_6"
                    elif task == 11:
                        c = "japanese_6"
                    elif task == 12:
                        c = "english_7"
                    elif task == 13:
                        c = "japanese_7"
                    elif task == 14:
                        c = "english_8"
                    elif task == 15:
                        c = "japanese_8"
                    elif task == 16:
                        c = "english_9"
                    elif task == 17:
                        c = "japanese_9"
                    elif task == 18:
                        c = "english_10"
                    elif task == 19:
                        c = "japanese_10"

                    if l == 0:
                        print("SGD" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))
                    else:
                        print("EWC" + " " + c + ":" +
                              str(test_accs[task][int(iter / disp_freq)]))

                    #plot_h, = plt.plot(range(1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], colors[task], label=c)

                    plot_h, = plt.plot(range(
                        1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1], label=c)

                    plots.append(plot_h)

                plot_test_acc(plots)

                if l == 0:
                    plt.title("Stochastic Gradient Descent")  # 確率的勾配降下法
                else:
                    plt.title("Elastic Weight Consolidation")

                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

    plt.show()


japanese_model_path = "/Users/kawahara/Documents/01-programming/00-大学研究/01-mailsearch/japanese/doc2vec_models/"
english_model_path = "/Users/kawahara/Documents/01-programming/00-大学研究/01-mailsearch/trec_doc2vec/trec_year_doc2vec_model/"

"""
日本語のdoc2vecの読み込み
japanese_doc2vec_1_list:前半の1000個がspam、後半の1000個がham
"""
japanese_doc2vec_1 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_1.model")
japanese_doc2vec_1_list = [japanese_doc2vec_1.docvecs[num]
                           for num in range(len(japanese_doc2vec_1.docvecs))]

japanese_doc2vec_2 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_2.model")
japanese_doc2vec_2_list = [japanese_doc2vec_2.docvecs[num]
                           for num in range(len(japanese_doc2vec_2.docvecs))]

japanese_doc2vec_3 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_3.model")
japanese_doc2vec_3_list = [japanese_doc2vec_3.docvecs[num]
                           for num in range(len(japanese_doc2vec_3.docvecs))]

japanese_doc2vec_4 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_4.model")
japanese_doc2vec_4_list = [japanese_doc2vec_4.docvecs[num]
                           for num in range(len(japanese_doc2vec_4.docvecs))]

japanese_doc2vec_5 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_5.model")
japanese_doc2vec_5_list = [japanese_doc2vec_5.docvecs[num]
                           for num in range(len(japanese_doc2vec_5.docvecs))]

japanese_doc2vec_6 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_6.model")
japanese_doc2vec_6_list = [japanese_doc2vec_6.docvecs[num]
                           for num in range(len(japanese_doc2vec_6.docvecs))]

japanese_doc2vec_7 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_7.model")
japanese_doc2vec_7_list = [japanese_doc2vec_7.docvecs[num]
                           for num in range(len(japanese_doc2vec_7.docvecs))]

japanese_doc2vec_8 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_8.model")
japanese_doc2vec_8_list = [japanese_doc2vec_8.docvecs[num]
                           for num in range(len(japanese_doc2vec_8.docvecs))]

japanese_doc2vec_9 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_9.model")
japanese_doc2vec_9_list = [japanese_doc2vec_9.docvecs[num]
                           for num in range(len(japanese_doc2vec_9.docvecs))]

japanese_doc2vec_10 = Doc2Vec.load(
    japanese_model_path + "japanese_doc2vec_10.model")
japanese_doc2vec_10_list = [japanese_doc2vec_10.docvecs[num]
                            for num in range(len(japanese_doc2vec_10.docvecs))]

"""
英語のdoc2vecの読み込み
前半の5000個がspam、後半の5000個が
"""
english_doc2vec_first_half = Doc2Vec.load(
    english_model_path + "2005_doc2vec.model")
english_doc2vec_first_half_list = [english_doc2vec_first_half.docvecs[num]
                                   for num in range(len(english_doc2vec_first_half.docvecs))]

english_doc2vec_latter_half = Doc2Vec.load(
    english_model_path + "2006_doc2vec.model")
english_doc2vec_latter_half_list = [english_doc2vec_latter_half.docvecs[num]
                                    for num in range(len(english_doc2vec_latter_half.docvecs))]

list_1 = list(range(1000))
list_1.extend(list(range(5000, 6000)))
english_doc2vec_1_list = [
    english_doc2vec_first_half_list[num] for num in list_1]

list_2 = list(range(1000, 2000))
list_2.extend(list(range(6000, 7000)))
english_doc2vec_2_list = [
    english_doc2vec_first_half_list[num] for num in list_2]

list_3 = list(range(2000, 3000))
list_3.extend(list(range(7000, 8000)))
english_doc2vec_3_list = [
    english_doc2vec_first_half_list[num] for num in list_3]

list_4 = list(range(3000, 4000))
list_4.extend(list(range(8000, 9000)))
english_doc2vec_4_list = [
    english_doc2vec_first_half_list[num] for num in list_4]

list_5 = list(range(4000, 5000))
list_5.extend(list(range(9000, 10000)))
english_doc2vec_5_list = [
    english_doc2vec_first_half_list[num] for num in list_5]

english_doc2vec_6_list = [
    english_doc2vec_latter_half_list[num] for num in list_1]

english_doc2vec_7_list = [
    english_doc2vec_latter_half_list[num] for num in list_2]

english_doc2vec_8_list = [
    english_doc2vec_latter_half_list[num] for num in list_3]

english_doc2vec_9_list = [
    english_doc2vec_latter_half_list[num] for num in list_4]

english_doc2vec_10_list = [
    english_doc2vec_latter_half_list[num] for num in list_5]

print(len(english_doc2vec_10_list[0]))

sess = tf.InteractiveSession()

mail_doc2vec = tf.placeholder(tf.float32, shape=[None, 300])
mail_class = tf.placeholder(
    tf.float32, shape=[None, 2])  # ham:[1,0],spam:[0,1]

model = Model(mail_doc2vec, mail_class)  # simple 2-layer network

sess.run(tf.global_variables_initializer())

# training 1st task
# 最期から二個目の引き数:言語の選択={0:english,1:japanese}
train_task(model, 320, 20, english_doc2vec_1_list, [
           english_doc2vec_1_list], mail_doc2vec, mail_class, 0, lams=[0])
print('train_task finished')

# Fisher information
# use validation set for Fisher computation
model.compute_fisher(english_doc2vec_1_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

# save current optimal weights
model.star()

test_ephoc = 0
train_ephoc = 0

print("english_1 task finished")

# training 2nd task
train_task(model, 320, 20, japanese_doc2vec_1_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list], mail_doc2vec, mail_class, 1, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_1_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_1 task finished")

# training 3nd task
train_task(model, 320, 20, english_doc2vec_2_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_2_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_2 task finished")

# training 4th task
train_task(model, 320, 20, japanese_doc2vec_2_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_2_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_2 task finished")

# training 5th task
train_task(model, 320, 20, english_doc2vec_3_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_3_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_3 task finished")

# training 6th task
train_task(model, 320, 20, japanese_doc2vec_3_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_3_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_3 task finished")

# training 7th task
train_task(model, 320, 20, english_doc2vec_4_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_4_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_4 task finished")

# training 8th task
train_task(model, 320, 20, japanese_doc2vec_4_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_4_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_4 task finished")

# training 9th task
train_task(model, 320, 20, english_doc2vec_5_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_5_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_5 task finished")

# training 10th task
train_task(model, 320, 20, japanese_doc2vec_5_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_5_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_5 task finished")

# training 11th task
train_task(model, 320, 20, english_doc2vec_6_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_6_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_6 task finished")

# training 12th task
train_task(model, 320, 20, japanese_doc2vec_6_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_6_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_6 task finished")

# training 13th task
train_task(model, 320, 20, english_doc2vec_7_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_7_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_7 task finished")

# training 14th task
train_task(model, 320, 20, japanese_doc2vec_7_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_7_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_7 task finished")

# training 15th task
train_task(model, 320, 20, english_doc2vec_8_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_8_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_8 task finished")

# training 16th task
train_task(model, 320, 20, japanese_doc2vec_8_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list, japanese_doc2vec_8_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_8_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_8 task finished")

# training 17th task
train_task(model, 320, 20, english_doc2vec_9_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list, japanese_doc2vec_8_list, english_doc2vec_9_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_9_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_9 task finished")

# training 18th task
train_task(model, 320, 20, japanese_doc2vec_9_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list, japanese_doc2vec_8_list, english_doc2vec_9_list, japanese_doc2vec_9_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_9_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_9 task finished")

# training 19th task
train_task(model, 320, 20, english_doc2vec_10_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list, japanese_doc2vec_8_list, english_doc2vec_9_list, japanese_doc2vec_9_list, english_doc2vec_10_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(english_doc2vec_10_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("english_10 task finished")

# training 20th task
train_task(model, 320, 20, japanese_doc2vec_10_list, [
           english_doc2vec_1_list, japanese_doc2vec_1_list, english_doc2vec_2_list, japanese_doc2vec_2_list, english_doc2vec_3_list, japanese_doc2vec_3_list, english_doc2vec_4_list, japanese_doc2vec_4_list, english_doc2vec_5_list, japanese_doc2vec_5_list, english_doc2vec_6_list, japanese_doc2vec_6_list, english_doc2vec_7_list, japanese_doc2vec_7_list, english_doc2vec_8_list, japanese_doc2vec_8_list, english_doc2vec_9_list, japanese_doc2vec_9_list, english_doc2vec_10_list, japanese_doc2vec_10_list], mail_doc2vec, mail_class, 2, lams=[0, 40])

model.compute_fisher(japanese_doc2vec_10_list, sess,
                     num_samples=200, plot_diffs=True)
print("model.compute_fisher finished")

model.star()

test_ephoc = 0
train_ephoc = 0

print("japanese_10 task finished")
