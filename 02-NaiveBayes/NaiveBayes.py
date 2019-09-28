import sys
import math
import numpy as np
import codecs
from collections import OrderedDict


class Bayesian_Filter:
    def __init__(self):
        self.vocabularies = set()  # 単語の集合
        self.wordcount = {}  # {spam:{word1:a個,word2:b個,...},ham:{word1:c個,word2:d個,...}}
        self.catcount = {}  # {spam:a通,ham:b通}

    # 単語のカウント
    def wordcountup(self, word, cat):
        self.wordcount.setdefault(cat, {})
        self.wordcount[cat].setdefault(word, 0)
        self.wordcount[cat][word] += 1
        self.vocabularies.add(word)  # 重複を除く

    # スパムメールの件数、正規メールの件数
    def catcountup(self, cat):
        self.catcount.setdefault(cat, 0)
        self.catcount[cat] += 1

    # 学習
    def train(self, doc, cat):
        word = doc.split()
        for w in word:
            self.wordcountup(w, cat)
        self.catcountup(cat)

    # spam中にwordが出現した回数,ham中にwordが出現した回数
    def incategory(self, word, cat):
        if word in self.wordcount[cat]:
            return float(self.wordcount[cat][word])
        return 0.0

    # Gary-Robinson方式
    def Gary_Robinson(self, doc):
        words = doc.split()  # 今回使用するメールデータ(特に日本語のメール)に関しては、すでに分かち書きを行っている
        F_W = 0
        F_W_for_P = 1
        F_W_for_Q = 1
        P = 0
        Q = 0
        S = 0
        mail_words_num = len(words)  # メール内の総単語数

        # まず、各単語のスパム確率を計算する.
        for word in words:
            word_count_spam = self.incategory(
                word, "spam")  # スパムメールでwordが出現した回数
            word_count_ham = self.incategory(word, "ham")  # 正規メールでwordが出現した回数
            word_count = word_count_ham + word_count_spam  # wordが出現した回数

            s_word = word_count_spam / self.catcount["spam"]
            h_word = word_count_ham / self.catcount["ham"]

            if (h_word + s_word) == 0:
                P_WORD = 0.4  # 未知後が来た場合0.4または0.415が妥当とされている
            else:
                P_WORD = s_word / (h_word + s_word)  # 単語wordのspam確率

            F_W = (0.5 + word_count * P_WORD) / (1 + word_count)

            F_W_for_P *= (1 - F_W)
            F_W_for_Q *= F_W

        P = 1 - math.pow(F_W_for_P, 1 / mail_words_num)
        Q = 1 - math.pow(F_W_for_Q, 1 / mail_words_num)

        S = (P - Q) / (P + Q)
        S = (1 + S) / 2

        if S > 0.45:
            return "spam"
        else:
            return "ham"

    # 識別
    # self, month, mail__text, mail_dict
    def classifier(self, month, ham_mail, spam_mail, train_id, test_id, mail_dict):
        spam_check_num = 0
        ham_check_num = 0

        ham_mail_path = open(ham_mail, 'r')
        spam_mail_path = open(spam_mail, 'r')

        ham_mails = ham_mail_path.readlines()
        spam_mails = spam_mail_path.readlines()

        ham_mail_path.close()
        spam_mail_path.close()

        # 訓練
        for mail_id in train_id:
            self.train(spam_mails[mail_id], 'spam')
        for mail_id in train_id:
            self.train(ham_mails[mail_id], 'ham')

        # 教師ラベルの登録
        for mail_id in test_id:
            spam_mail = spam_mails[mail_id]
            ham_mail = ham_mails[mail_id]

            mail_dict[month][spam_mail] = 'spam'
            mail_dict['all_language'][spam_mail] = 'spam'

            mail_dict[month][ham_mail] = 'ham'
            mail_dict['all_language'][ham_mail] = 'ham'

        for checked_month in mail_dict.keys():
            if checked_month == 'all_language' and len(mail_dict) == 2:
                continue

            category_list = list(mail_dict[checked_month].values())

            spam_mail_num = category_list.count('spam')
            ham_mail_num = category_list.count('ham')

            for mail_body, cat in mail_dict[checked_month].items():  # 推定
                # 判定が正しかった場合
                if cat == bayesian_filter.Gary_Robinson(mail_body):
                    if cat == "spam":
                        spam_check_num += 1
                    else:
                        ham_check_num += 1

            spam_accuracy = (spam_check_num / spam_mail_num) * 100
            ham_accuracy = (ham_check_num / ham_mail_num) * 100
            all_accuracy = (spam_check_num+ham_check_num) / \
                (spam_mail_num+ham_mail_num)*100

            print('{} spam accuracy : {:.2f} %'.format(
                checked_month, spam_accuracy))
            print('{} ham accuracy : {:.2f} %'.format(
                checked_month, ham_accuracy))
            print('all accuracy : {:.2f} %'.format(all_accuracy))

            spam_check_num = 0
            ham_check_num = 0

        print("#" * 50)


if __name__ == '__main__':
    mail_path = "/Users/kawahara/Documents/10-maildata/01-spam/spam_mails/"
    bayesian_filter = Bayesian_Filter()

    ham_mail = mail_path + "ham_at_May_wakati.txt"
    spam_mail = mail_path + "spam_at_May_wakati.txt"

    mail_dict = OrderedDict()
    mail_dict['all_language'] = dict()

    while True:
        # Mar:March 三月
        # A:April 四月
        # May:五月
        # R :リセット
        # F:終了
        learn_language = input('Mar ,A, May , R or F : ')

        if learn_language == 'Mar':
            train_id = range(1000)
            test_id = range(1000, 2000)

            mail_dict['Mar'] = dict()
            bayesian_filter.classifier(
                'Mar', ham_mail, spam_mail, train_id, test_id, mail_dict)

        elif learn_language == 'A':
            train_id = range(2000, 3000)
            test_id = range(3000, 4000)

            mail_dict['April'] = dict()
            bayesian_filter.classifier(
                'April', ham_mail, spam_mail, train_id, test_id, mail_dict)

        elif learn_language == 'May':
            train_id = range(4000, 5000)
            test_id = range(5000, 5700)

            mail_dict['May'] = dict()
            bayesian_filter.classifier(
                'May', ham_mail, spam_mail, train_id, test_id, mail_dict)

        elif learn_language == 'R':
            mail_dict.clear()
            mail_dict['all_language'] = dict()
            print('初期化　完了')

        elif learn_language == 'F':
            print('終了')
            break
