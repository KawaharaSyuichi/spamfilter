"""
全ての年でdoc2vecを作成
"""

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer
import tkinter
import gensim
import csv
import re

PATH = "/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_mail_set_by_2020/all_year/all_year"
MODEL_SAVE_PATH = "/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_year_doc2vec_model/"


def make_doc2vec(mail_path):
    with open(mail_path, "r") as f:
        print("mail file read finished\n")

        mail_trainings = [TaggedDocument(words=data.split(), tags=[
            i]) for i, data in enumerate(f)]

        print("mails TaggedDocument finished\n")

        model = Doc2Vec(documents=mail_trainings, dm=1, vector_size=300,
                        windows=5, min_count=1, epochs=400, workers=4)

        print("make Doc2vec finished\n")

        model.save(MODEL_SAVE_PATH + "all_year_doc2vec.model")


def change_doc2vec_model_to_csv(model_path):
    model = Doc2Vec.load(model_path + "all_year_doc2vec.model")

    doc2vec_lists = [model.docvecs[num] for num in range(len(model.docvecs))]
    with open(model_path  + "all_year_doc2vec.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(doc2vec_lists)


make_doc2vec(PATH)
change_doc2vec_model_to_csv(MODEL_SAVE_PATH)
