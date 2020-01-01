# t-SNEの発音は「ティースニィー」でOK?
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from gensim.models import Doc2Vec


def t_SEN_plot_same_language(doc2vec_model_file):
    doc2vec_model = Doc2Vec.load(doc2vec_model_file)
    doc2vec_array_data = [doc2vec_model.docvecs[num]
                          for num in range(len(doc2vec_model.docvecs))]

    doc2vec_list_data = []
    for doc2vec_data in doc2vec_array_data:
        # doc2vecのデータをarrayからlistに変換
        doc2vec_list_data.append(doc2vec_data.tolist())

    # 今回使用するdoc2vecのモデルは，前半の5千個がスパムメール，残りの5千個が正規メールのdoc2vecとなっている
    # スパムメールをのラベルを0，正規メールのラベルを1とする
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
        doc2vec_list_data)

    plt.scatter(X_reduced[:5000, 0], X_reduced[:5000, 1],
                c="none", label="spam", edgecolors="red", linewidth=0.5)
    plt.scatter(X_reduced[5000:, 0], X_reduced[5000:, 1],
                c="none", label="ham", edgecolors="blue", linewidth=0.5)

    plt.legend(loc="lower right")
    plt.show()


def t_SEN_plot_same_mail_type(doc2vec_array_data, mail_class):

    doc2vec_list_data = []
    for doc2vec_data in doc2vec_array_data:
        # doc2vecのデータをarrayからlistに変換
        doc2vec_list_data.append(doc2vec_data.tolist())

    # 今回使用するdoc2vecのモデルは，前半の千個が英語メール，残りの千個が日本語メールのdoc2vecとなっている
    # 英語メールのラベルを0，日本語のラベルを1とする
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
        doc2vec_array_data)

    plt.scatter(X_reduced[:5000, 0], X_reduced[:5000, 1],
                c="none", label="2005 " + mail_class, edgecolors="red", linewidth=0.5)
    plt.scatter(X_reduced[5000:10000, 0], X_reduced[5000:10000, 1],
                c="none", label="2006 " + mail_class, edgecolors="blue", linewidth=0.5)
    plt.scatter(X_reduced[10000:15000, 0], X_reduced[10000:15000, 1],
                c="none", label="2007 " + mail_class, edgecolors="green", linewidth=0.5)

    plt.legend(loc="lower right")
    # plt.colorbar()
    plt.show()


def t_SEN_plot_all(doc2vec_array_data):
    doc2vec_list_data = []
    for doc2vec_data in doc2vec_array_data:
        # doc2vecのデータをarrayからlistに変換
        doc2vec_list_data.append(doc2vec_data.tolist())

    # 英語のスパムメール，英語の正規メール，日本語のスパムメール，日本語の正規メール
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
        doc2vec_array_data)

    plt.scatter(X_reduced[:5000, 0], X_reduced[:5000, 1],
                c="none", label="2005 spam", edgecolors="red", linewidth=0.5)
    plt.scatter(X_reduced[5000:10000, 0], X_reduced[5000:10000, 1],
                c="none", label="2005 ham", edgecolors="blue", linewidth=0.5)

    plt.scatter(X_reduced[10000:15000, 0], X_reduced[10000:15000, 1],
                c="none", label="2006 spam", edgecolors="green", linewidth=0.5)
    plt.scatter(X_reduced[15000:20000, 0], X_reduced[15000:20000, 1],
                c="none", label="2006 ham", edgecolors="pink", linewidth=0.5)

    plt.scatter(X_reduced[20000:25000, 0], X_reduced[20000:25000, 1],
                c="none", label="2007 spam", edgecolors="purple", linewidth=0.5)
    plt.scatter(X_reduced[25000:30000, 0], X_reduced[25000:30000, 1],
                c="none", label="2007 ham", edgecolors="black", linewidth=0.5)

    plt.legend(loc="lower right")
    plt.show()


def main():
    # 読み込むdoc2vecモデルのリスト
    model_file_list = ["trec_year_doc2vec_model/2005_doc2vec.model",
                       "trec_year_doc2vec_model/2006_doc2vec.model",
                       "trec_year_doc2vec_model/2007_doc2vec.model"]

    doc2vec_spam_array_data = []
    doc2vec_ham_array_data = []

    for doc2vec_model_file in model_file_list:
        # 同じ言語のスパムメールと正規メールの分布を求める
        t_SEN_plot_same_language(doc2vec_model_file)

    for doc2vec_model_file in model_file_list:
        doc2vec_model = Doc2Vec.load(doc2vec_model_file)
        all_vector = [doc2vec_model.docvecs[num]
                      for num in range(len(doc2vec_model.docvecs))]
        doc2vec_spam_array_data.extend(all_vector[:5000])  # spam
        doc2vec_ham_array_data.extend(all_vector[5000:])  # ham

    t_SEN_plot_same_mail_type(doc2vec_spam_array_data, "spam")
    t_SEN_plot_same_mail_type(doc2vec_ham_array_data, "ham")

    doc2vec_array_data = []
    for doc2vec_model_file in model_file_list:
        doc2vec_model = Doc2Vec.load(doc2vec_model_file)
        all_vector = [doc2vec_model.docvecs[num]
                      for num in range(len(doc2vec_model.docvecs))]
        doc2vec_array_data.extend(all_vector)

    t_SEN_plot_all(doc2vec_array_data)


if __name__ == "__main__":
    main()
