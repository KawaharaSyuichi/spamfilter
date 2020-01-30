# プログラムの内容
破滅的忘却を軽減するニューラルネットワークを用いたスパムフィルタ

## 破滅的忘却
近年、ニューラルネットワークを用いた文章分類が活発に行われており、スパムフィルタへの応用が期待されています。 

しかし、学習済みのニューラルネットワークに、新たなデータBを追加学習させると、その新たなデータに対する正解率は上がるが、
以前に学習したデータAに対する正解率は低下してしまいます。 

これは、ニューラルネットワークが以前のデータAを学習したときに得られたパラメータ(重みやバイアス)の値は、
そのデータAの正解率を上げることに適したパラメータになっているため、そこの新たなデータBを追加学習させると、
ニューラルネットワークのパラメータは新たなデータBの正解率を上げるためのパラメータへと更新されてしまうため、
以前に学習したデータAに対しての正解率が低下してしまうためです。 

このように、ニューラルネットワークに新たなデータを追加学習させると、以前のデータを学習して得られたパラメータを忘却してしまう(新たなデータに適したパラメータに更新される)ことを**破滅的忘却**といいます。

# 既存手法
ニューラルネットワークの課題の一つである**破滅的忘却**を解決する既存手法としては、学習させたい新しいデータがある場合、過去のデータを学習したときに得られたニューラルネットワークのパラメータは捨てて、過去のデータとその新しいデータをまとめて新たにニューラルネットワークに学習させることで、過去のデータと新しいデータの両方のデータに対する正解率を算出するという手法をとっていました。

しかし、この手法では以下の問題点が挙げられます。
- 過去に学習したデータを保持し続ける必要がある
- 学習時間が増加する

こういった問題点があるなか、近年、過去に学習したデータを保持せずに新たなデータを追加学習させても、破滅的忘却を軽減できる**Elastic** **Weight** **Consolidation**(**EWC**)と呼ばれる手法が注目を集めています。

## Elastic Weight Consolidation(EWC)
Elastic Weight Consolidationでは、過去に学習したデータの正解率を保持するために重要なパラメータの要素(例えば重みの何行何列目の要素など)を確率的に導き出し、新たなデータを学習するときに、そのパラメータの要素の更新量を抑えことで、過去のデータに対する正解率をなるべく保持しつつ、新たなデータに対する正解率も上げることができる。

ニューラルネットワークの出力値と正解データの誤差を計算し，その誤差が最小になるように重みやバイアスの更新を行う．
誤差の計算方法として，テキスト分類等のクラス分類では交差エントロピーと呼ばれる損失関数が一般的に使用され，以下の式で計算される．

<img src="https://latex.codecogs.com/gif.latex?L=-\sum_{i=1}^{n}t_i\log&space;y_i"/>

ここで，<img src="https://latex.codecogs.com/gif.latex?y_i">は出力層のi番目の出力値，<img src="https://latex.codecogs.com/gif.latex?t_i">は教師ラベルで，入力されたデータのクラス情報が含まれている．

損失を小さくするための重みの更新方法として様々な手法が存在する．ここでは，本研究でも使用する確率的勾配降下法について説明する．

確率的勾配降下法では，損失関数Lを用いて以下の式に示すように，重みを更新する．

<img src="https://latex.codecogs.com/gif.latex?W^{'}&space;\leftarrow&space;W-\eta&space;\frac{\partial&space;L}{\partial&space;W}">

ここで，<img src="https://latex.codecogs.com/gif.latex?W^{'}">は更新後の重み，<img src="https://latex.codecogs.com/gif.latex?W">は更新前の重みを表している．また，<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;W}" width="10" height="10">は重み<img src="https://latex.codecogs.com/gif.latex?W">に関する損失関数の勾配，<img src="https://latex.codecogs.com/gif.latex?\eta">は学習係数を表しており，重みの更新量を制御するための係数である．

EWCを用いたニューラルネットワークで追加学習を行う場合，損失関数Lを以下の式で計算する．

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta)=\mathcal{L}_{B}(\theta)&plus;\frac{\lambda}{2}\sum_{i}F_{i}(\theta_{i}-\theta_{A,i}^{*})^{2}">

上式の右辺の<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{B}(\theta)">は新たに学習するデータBの追加学習を行う場合に得られる損失であり，EWCではこの損失<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{B}(\theta)">に<img src="https://latex.codecogs.com/gif.latex?\frac{\lambda}{2}\sum_{i}F_{i}(\theta_{i}-\theta_{A,i}^{*})^{2}">を加算する．
<img src="https://latex.codecogs.com/gif.latex?F_{i}">はデータAに対するパラメータ(重みとバイアス)のフィッシャー情報行列， <img src="https://latex.codecogs.com/gif.latex?\theta_{i}">がデータBの学習時に使用するパラメータ， <img src="https://latex.codecogs.com/gif.latex?\theta_{A,i}^{*}">がデータAの学習時に使用したパラメータを表している． <img src="https://latex.codecogs.com/gif.latex?\frac{\lambda}{2}\sum_{i}F_{i}(\theta_{i}-\theta_{A,i}^{*})^{2}">には，以前に学習したデータAを学習して得られたパラメータのうち，データAを識別するために重要なパラメータの要素の情報を含んでいる．

EWCでは式<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta)">で計算された損失の値を小さくなるように学習を行い，メールのデータAに対して重要なパラメータの要素の更新を遅らせることで，過去に学習した情報の忘却を軽減する．

# 各フォルダの説明
・  
|  
|_01-IMAP:メールサーバからメールを受信するためのプログラム  
|_02-NaiveBayes:ナイーブベイズを用いたスパムフィルタのプログラム  
|_03-SpamFilter_by_NeuralNetwork_using_EWC：私の研究に関するプログラム  
&nbsp;&nbsp; |_doc2vec:doc2vecで生成したベクトルとそのベクトルをt-SNEを用いて二次元の画像にプロットした画像  
&nbsp;&nbsp; |_EWC:EWCを用いたデフォルトのスパムフィルタのプログラム  
&nbsp;&nbsp; |_EWC_hyperparameter:中間層の数や各層のユニット数等のハイパーパラメータの最適な組み合わせを探索するプログラム  
&nbsp;&nbsp; |_EWC_with_GUI:EWCを用いたデフォルトのプログラムをGUIで操作できるようにしたプログラム  


# 論文 URL
[論文説明](https://www.ieice.org/ken/paper/20190723N1Of/)

# 学会　受賞歴
[SITE学術奨励賞　2019年7月　「破滅的忘却を軽減するニューラルネットワークを用いたスパムフィルタの提案」](https://www.ieice.org/~site/site_award.html)