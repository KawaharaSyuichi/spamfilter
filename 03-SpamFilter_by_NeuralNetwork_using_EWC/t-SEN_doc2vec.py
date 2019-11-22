import sys
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from gensim.models import Doc2Vec


def t_SEN_plot(model_file):
    english_doc2vec = Doc2Vec.load(model_file)
    english_doc2vec_array_data = [english_doc2vec.docvecs[num]
                                  for num in range(len(english_doc2vec.docvecs))]

    english_doc2vec_list_data = []
    for doc2vec_array_data in english_doc2vec_array_data:
        english_doc2vec_list_data.append(doc2vec_array_data.tolist())

    english_doc2vec_target = [0] * 1000
    english_doc2vec_target.extend([1] * 1000)

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
        english_doc2vec_list_data)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=english_doc2vec_target)
    plt.colorbar()
    plt.show()


model_file_list = ["doc2vec_english.model", "doc2vec_japanese.model"]
for doc2vec_file in model_file_list:
    t_SEN_plot(doc2vec_file)
