import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython import display


def weight_variable(shape):
    """
    重みの初期化
    """
    # shape分の各要素が正規分布かつ標準偏差の2倍までのランダムな値で初期化
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    バイアスの初期化
    """
    initial = tf.constant(0.1, shape=shape)  # shape分の各要素が0.1の行列
    return tf.Variable(initial)


class Model:
    def __init__(self, x, y_):
        """
        ニューラルネットワークモデルの構築
        入力層のユニット数：768
        中間層のユニット数：100
        出力層のユニット数：2
        中間層の層数：1
        使用する活性化関数：ReLU関数
        使用する損失関数：交差エントロピー損失関数
        """

        in_dim = 768  # bertを用いて得られたベクトルの次元数
        out_dim = 2  # スパムメールと正規メールの2クラス

        self.x = x  # input placeholder (x:[None,768])

        # 入力層と中間層の間のパラメータ
        W1 = weight_variable([in_dim, 128])
        b1 = bias_variable([128])

        # 中間層と出力層の間のパラメータ
        W2 = weight_variable([128, out_dim])
        b2 = bias_variable([out_dim])

        # ReLU関数：活性化関数の一つで、入力した値が0以下のときに0、0より大きい場合はそのままの値を出力する関数
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer
        self.y = tf.matmul(h1, W2) + b2  # output layer

        self.var_list = [W1, b1, W2, b2]

        # 交差エントロピー損失関数
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def compute_fisher(self, doc2vec_model, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        """
        フィッシャー情報量の計算
        """

        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(
                np.zeros(self.var_list[v].get_shape().as_list()))

        # 最も確率の高いクラスを選択
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.argmax(tf.log(probs), 1)[0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        sum_validation = []
        sum_validation.extend(list(range(1000, 1100)))  # [1000,1001,...,1099]
        sum_validation.extend(list(range(100)))  # [0,1,2,...,99]

        for i in range(num_samples):  # num_samples=200
            # フィッシャー情報行列の計算
            im_ind = sum_validation.pop(0)

            # 一次微分を計算する
            # tf.gradientsを実行(sess.run)するうえで、スペースホルダーに該当する部分に、feed_dictで指定した値を使用する。
            ders = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list), feed_dict={
                            self.x: np.array(doc2vec_model[im_ind]).reshape(1, 768)})

            for v in range(len(self.F_accum)):
                # 求めた勾配の二乗を該当するフィッシャー情報行列に加算していく。
                self.F_accum[v] += np.square(ders[v])

            if (plot_diffs):  # フィッシャー情報量のプロット処理
                if i % disp_freq == 0 and i > 0:  # disp_freq=10
                    # recording mean diffs of F
                    F_diff = 0

                    for v in range(len(self.F_accum)):
                        # np.absolute:絶対値を求める。
                        F_diff += np.sum(np.absolute(
                            self.F_accum[v] / (i + 1) - F_prev[v]))

                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)

                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v] / (i + 1)

                    plt.plot(range(disp_freq + 1, i + 2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples  # 平均を求める

        plt.show()

    def stor(self):
        """
        パラメータ(重みとバイアス)の保存
        """
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        """
        保存していたパラメータの読み込み
        """
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        """
        Stochastic Gradient Descent(SGD)を用いてパラメータを更新
        """
        self.train_step = tf.compat.v1.train.GradientDescentOptimizer(
            0.1).minimize(self.cross_entropy)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous language_task constraints
        """
        Elastic Weight Consolidationを用いてパラメータを更新
        lam : 一つ前に学習したデータに対する重みづけ(過去のデータをどれだけ考慮するかを決めるパラメータ)
        """

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        # フィッシャー情報量の計算
        for v in range(len(self.var_list)):
            fisher = (lam / 2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(
                np.float32), tf.square(self.var_list[v] - self.star_vars[v])))

            self.ewc_loss += fisher

        # フィッシャー情報量を用いてパラメータ更新
        self.train_step = tf.train.GradientDescentOptimizer(
            0.1).minimize(self.ewc_loss)  # 学習率0.1として、ewcで求めた損失関数値を最小にするようにパラメータを学習する。
