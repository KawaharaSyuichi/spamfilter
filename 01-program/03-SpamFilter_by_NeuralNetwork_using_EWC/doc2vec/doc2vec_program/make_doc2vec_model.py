"""
各年のメール2000通分(1000通分がスパムメール、1000通文が正規メール)のdoc2vecモデルを作成
"""
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

import tkinter
import gensim
import re
import csv

index_year = ["2005", "2006", "2007"]
common_path = '/Users/kawahara/Documents/01-programming/00-大学研究/02-result' \
              '/trec_doc2vec/trec_mail_set_by_2020/'
doc2vec_save_path = '/Users/kawahara/Documents/01-programming/00-大学研究/02' \
                    '-result/trec_doc2vec/trec_year_doc2vec_model/'


def make_doc2vec(mail_path, year):
    with open(mail_path + year + '/' + year + '_2000_mails', "r") as f:
        print("mail file read finished\n")

        mail_trainings = [TaggedDocument(words=data.split(), tags=[i]) for
                          i, data in enumerate(f)]

        print("mails TaggedDocument finished\n")

        model = Doc2Vec(documents=mail_trainings, dm=1, vector_size=300,
                        windows=5, min_count=1, epochs=400, workers=4)

        print(year + "make Doc2vec finished\n")

        model.save(doc2vec_save_path + year + "_2000_mails_doc2vec.model")


def change_doc2vec_model_to_csv(model_path, mail_set_path):
    model = Doc2Vec.load(model_path + mail_set_path)

    doc2vec_lists = [model.docvecs[num] for num in range(len(model.docvecs))]
    with open(model_path + "_2000_mails_doc2vec.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(doc2vec_lists)


for year in index_year:
    make_doc2vec(common_path, year)
    change_doc2vec_model_to_csv(doc2vec_save_path,
                                str(year) + '_2000_mails_doc2vec.model')

"""
for year in index_year:
    with open(common_path + year + '/' + year + '_all_spam', 'r') as spam_f:
        spam_mail_bodies = spam_f.readlines()

    with open(common_path + year + '/' + year + '_2000_mails',
              'a') as new_spam_f:
        for i, spam_mail_body in enumerate(spam_mail_bodies):
            new_spam_f.write(spam_mail_body)

            if i == 999:
                break

    with open(common_path + year + '/' + year + '_all_ham', 'r') as ham_f:
        ham_mail_bodies = ham_f.readlines()

    with open(common_path + year + '/' + year + '_2000_mails',
              'a') as new_ham_f:
        for i, ham_mail_body in enumerate(ham_mail_bodies):
            new_ham_f.write(ham_mail_body)

            if i == 999:
                break
"""
