# 年単位でdoc2vecを作成

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

import tkinter
import gensim
import re

ham_index = []
spam_index = []
mail_index = []
index_year = ["2005", "2006", "2007"]

pattern = ".*?(/.*)"
repattern = re.compile(pattern)


def make_trec_mail_set(mail_index, year):
    read_mail_num = 0
    for file_path in mail_index:
        with open("trec" + year + "/" + file_path, "r") as f:
            try:
                mail_content = f.read()
                read_mail_num += 1
            except:
                continue
            mail_content = ' '.join(mail_content.splitlines())  # 改行だけ削除

        with open("trec_mail_set/" + "trec_" + year, "a") as f:
            f.write(mail_content + "\n")

        if read_mail_num == 5000:
            break


def make_doc2vec(mail_path, year):
    with open(mail_path + year, "r") as f:
        print("mail file read finished\n")

        mail_trainings = [TaggedDocument(words=data.split(), tags=[
            i]) for i, data in enumerate(f)]

        print("mails TaggedDocument finished\n")

        model = Doc2Vec(documents=mail_trainings, dm=1, vector_size=300,
                        windows=5, min_count=1, epochs=400, workers=4)

        print(year + "make Doc2vec finished\n")

        model.save("trec_year_doc2vec_model/" + year + "_doc2vec.model")


for year in index_year:
    with open("trec_index/index_" + year, "r") as f:
        indexs = f.readlines()

    for index in indexs:
        result = repattern.findall(index)

        if index[0] == "s":
            spam_index.append(result[0])
        else:
            ham_index.append(result[0])

    make_trec_mail_set(spam_index, year)
    make_trec_mail_set(ham_index, year)

    # リストの初期化
    spam_index = []
    ham_index = []
    mail_index = []

for year in index_year:
    make_doc2vec("trec_mail_set/trec_", year)
