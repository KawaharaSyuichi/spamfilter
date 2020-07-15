"""
月単位でdoc2vecを作成
"""

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer
import tkinter
import gensim
import csv
import re


def make_doc2vec(mail_path, mail_set_path):
    with open(mail_path + mail_set_path, "r") as f:
        print("mail file read finished\n")

        mail_trainings = [TaggedDocument(words=data.split(), tags=[
                                         i]) for i, data in enumerate(f)]

        print("mails TaggedDocument finished\n")

        model = Doc2Vec(documents=mail_trainings, dm=1, vector_size=300,
                        windows=5, min_count=1, epochs=400, workers=4)

        print(mail_set_path + "make Doc2vec finished\n")

        model.save("../trec_doc2vec_model_over_year/" +
                   mail_set_path + "_doc2vec.model")


def change_doc2vec_model_to_csv(model_path, mail_set_path):
    model = Doc2Vec.load(
        model_path + mail_set_path + "_doc2vec.model")

    doc2vec_lists = [model.docvecs[num] for num in range(len(model.docvecs))]
    with open(model_path + mail_set_path + "_doc2vec.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(doc2vec_lists)


MAIL_SET = ("mail_set_1", "mail_set_2")

for mail_set_path in MAIL_SET:
    #make_doc2vec("../trec_mail_set_by_2020/collect_mail_set/", mail_set_path)

    change_doc2vec_model_to_csv(
        "../trec_doc2vec_model_over_year/", mail_set_path)
