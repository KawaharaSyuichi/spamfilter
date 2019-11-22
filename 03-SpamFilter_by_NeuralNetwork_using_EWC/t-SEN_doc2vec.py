import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from gensim.models import Doc2Vec


def t_SEN_plot(doc2vec_model_file):
    doc2vec_model = Doc2Vec.load(doc2vec_model_file)
    doc2vec_array_data = [doc2vec_model.docvecs[num]
                          for num in range(len(doc2vec_model.docvecs))]

    doc2vec_list_data = []
    for doc2vec_data in doc2vec_array_data:
        # doc2vecのデータをarrayからlistに変換
        doc2vec_list_data.append(doc2vec_data.tolist())

    # 今回使用するdoc2vecのモデルは，前半の千個がスパムメール，残りの千個が正規メールのdoc2vecとなっている
    # スパムメールをのラベルを0，正規メールのラベルを1とする
    doc2vec_target = [0] * 1000
    doc2vec_target.extend([1] * 1000)

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
        doc2vec_list_data)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=doc2vec_target)
    plt.colorbar()
    plt.show()


# 読み込むdoc2vecモデルのリスト
model_file_list = ["doc2vec_english.model", "doc2vec_japanese.model"]

for doc2vec_model_file in model_file_list:
    t_SEN_plot(doc2vec_model_file)
