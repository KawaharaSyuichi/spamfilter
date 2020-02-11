"""
Doc2vecを用いてメールに記載された文章の特徴を抽出したベクトルを生成するコード
ライブラリgensimを使用
"""

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

mails_path = "MAILS PATH"  # メールの文章を記載したファイルのパス
Mails_data = open(mails_path, 'r')

# 各メールに対してタグ付けを行う
Mails_trainings = [TaggedDocument(
    words=data.split(), tags=[i]) for i, data in enumerate(Mails_data)]

print("TaggedDocument finished\n")

# Doc2vecを用いて各メールに記載された文章をベクトルに変換
model = Doc2Vec(documents=Mails_trainings, dm=1, size=300,
                windows=5, min_count=1, epochs=400, workers=4)

print("Make Doc2vec finished\n")

model.save("doc2vec.model")  # doc2vec.modelという名前で作成したベクトルを保存
Mails_data.close()
