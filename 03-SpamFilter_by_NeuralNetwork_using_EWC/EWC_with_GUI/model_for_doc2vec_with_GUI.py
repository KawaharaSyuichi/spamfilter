import tensorflow as tf
import numpy as np


def weight_variable(shape):
    # shape分の各要素が正規分布かつ標準偏差の2倍までのランダムな値で初期化
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # shape分の各要素が0.1の行列
    return tf.Variable(initial)


class Model:
    def __init__(self, x, y_):

        in_dim = 300  # 300 for doc2vec
        out_dim = 2  # spam or ham

        self.x = x  # input placeholder (x:[None,300])

        # simple 2-layer network
        W1 = weight_variable([in_dim, 100])
        b1 = bias_variable([100])

        W2 = weight_variable([100, out_dim])
        b2 = bias_variable([out_dim])

        # ReLU関数：活性化関数の一つで、入力した値が0以下のときに0、0より大きい場合はそのままの値を出力する関数。
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer
        self.y = tf.matmul(h1, W2) + b2  # output layer

        self.var_list = [W1, b1, W2, b2]

        # vanilla single-task loss
        # 交差エントロピー損失関数
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        """
         tf.equalはtf.argmax(self.y,1)=[1,2,3],tf.argmax(y_,1)=[2,2,3]とすると
         tf.equalの戻り値は[False,True,True]となる。
        """

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))

        """
         tf.castは,correct_predictionの各要素[False,Ture,Ture]を二つ目の式数の型に変換する[0.,1.,1.,]
         tf.reduce_meanは平均を計算する。[0.,1.,1.]の場合、2/3=0.666667のように計算される。
        """
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def compute_fisher(self, doc2vec_model, sess, num_samples=200, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(
                np.zeros(self.var_list[v].get_shape().as_list()))

        # 最も確率の高いクラスを選択
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.argmax(tf.log(probs), 1)[0])

        sum_validation = []
        sum_validation.extend(list(range(200)))  # [0,1,2,...,99]

        for _ in range(num_samples):  # num_samples=200
            # フィッシャー情報行列の計算
            im_ind = sum_validation.pop(0)

            # compute first-order derivatives(derivative:導関数)
            # (訳:一次微分を計算する)
            # tf.gradientsを実行(sess.run)するうえで、スペースホルダーに該当する部分に、feed_dictで指定した値を使用する。
            ders = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list), feed_dict={
                            self.x: np.array(doc2vec_model[im_ind]).reshape(1, 300)})

            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                # 求めた勾配の二乗を該当するフィッシャー情報行列に加算していく。
                self.F_accum[v] += np.square(ders[v])

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples  # 平均を求める

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(
            0.1).minimize(self.cross_entropy)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous language_task constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            fisher = (lam / 2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(
                np.float32), tf.square(self.var_list[v] - self.star_vars[v])))

            self.ewc_loss += fisher

        self.train_step = tf.train.GradientDescentOptimizer(
            0.001).minimize(self.ewc_loss)  # 学習率0.1として、ewcで求めた損失関数値を最小にするようにパラメータを学習する。
