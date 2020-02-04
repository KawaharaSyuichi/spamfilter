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

# 提案手法
 1. 単語の抽出
 <div align="center"><img src="https://user-images.githubusercontent.com/26127488/73605992-97d0e380-45e8-11ea-8f62-87af010dfdde.jpg"></div>
各メールに記載された本文やメールヘッダから、単語を抽出します。
<br>
<br>
<br>

 2. TF/IDFによる事前処理
 <div align="center"><img src="https://user-images.githubusercontent.com/26127488/73605993-97d0e380-45e8-11ea-943a-ad41873a991c.jpg"></div>
TF/IDFを用いて、スパムメールにも正規のメールにも頻出する単語を削除します。なぜならば、スパムメールにも正規のメールにも頻出する単語は、スパムメールと正規のメールを識別するための情報としては価値が低いと判断したためです。実際に事前処理を行わなかったデータをニューラルネットワークに学習させた場合、識別率は8割程度となりましたが、事前処理を行ったデータをニューラルネットワークに学習させた場合、識別率が9割程度になりました。

また、次の手法で使用するDoc2vecを用いて高次元のベクトルを生成するときに、メールに含まれる単語の数が多いと、ベクトルの生成時間が長くなってしまいます。そのため、この事前処理の段階で単語の数を減らしておくことでベクトルのデータ生成にかかる時間を短縮します。
<br>
<br>
<br>

 3. Doc2vecによる分散表現
 <div align="center"><img src="https://user-images.githubusercontent.com/26127488/73605994-97d0e380-45e8-11ea-88b6-5e62c69d0cdd.jpg"></div>
ニューラルネットワークに学習させるデータは、数字である必要があるため、メールに記述された単語を直接使用することはできません。
そのため、文章に含まれる単語の種類や、単語が記述される順番の情報を元にその文章の特徴を高次元のベクトルに変換するDoc2vecと呼ばれる手法を用います。
Doc2vecで得られたベクトルをニューラルネットワークに学習させます。

- Doc2vecの参考資料
  - [Doc2vecの原論](https://arxiv.org/abs/1405.4053)
  - [ライブラリgensimを用いたDoc2vecの実装方法](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py)
<br>
<br>
<br>

 4. メールデータセットAを学習(初学習の場合)
 <div align="center"><img src="https://user-images.githubusercontent.com/26127488/73605995-98697a00-45e8-11ea-97d8-3ce6d4788e24.jpg" width="50%" height="50%"></div>
初学習の場合、収集したメールデータセットAに含まれるa通のメールをDoc2vecを用いてベクトルに変換する。
上の図では、m1が1つ目のメールのベクトル、m2が二つ目のメールのベクトル、maがa通目のメールのベクトルとなっている。こうして得られたベクトルの内、いくつかのベクトルを学習用データとしてニューラルネットワークで学習し、残りのベクトルをテスト用データとして使用し、識別率を求める。
このとき、初学習のためEWCを用いずに学習する。(EWCは過去に学習したデータに対する識別率の低下を抑える手法であるため、初学習の場合にはEWCを用いる必要がないため)
<br>
<br>
<br>

 5. メールデータセットBを追加学習
 <div align="center"><img src="https://user-images.githubusercontent.com/26127488/73605997-98697a00-45e8-11ea-8215-bffb1e44a558.jpg" width="50%" height="50%"></div>
メールデータセットAを学習後にメールデータセットBを追加学習する場合、収集したメールデータセットBに含まれるb通のメールをDoc2vecを用いてベクトルに変換する。 上の図では、m1が1つ目のメールのベクトル、m2が二つ目のメールのベクトル、mbがb通目のメールのベクトルとなっている。このとき、追加学習を行うため、EWCを用いて学習を行う。こうすることで、メールメータセットAに対する識別率の低下を軽減しつつ、メールデータセットBに対する識別率を上げることができる。なお、新たにメールデータセットC,D,E,...を追加学習する場合にも、EWCを用いて学習を行うこととなる。繰り返しになるが、EWCを用いないのは初学習の場合のみであり、なぜならば、EWCは過去に学習したデータに対する識別率の低下を軽減する手法であるためである。


# 実験結果
ここでは、TREC( https://trec.nist.gov/data/spam.html )で提供されている2005年〜2007年までの三年間のメールを一年間隔で学習させた場合の実験結果を示す。  

**実験条件**  
 - 学習させる順番：2005年→2006年→2007年の順
 - 各年の迷惑メールの内訳：10000通(半分を学習用、残り半分をテスト用として使用)
 - 各年の正規のメールの内訳：10000通(半分を学習用、残り半分をテスト用として使用)
 - バッチ数:80通(一度に学習するメールの数)
 - Doc2vecによる次元数：300
<br>
<br>

<div align="center"><img src="https://github.com/KawaharaSyuichi/spamfilter/blob/master/03-SpamFilter_by_NeuralNetwork_using_EWC/EWC/result/SGD_and_EWC_result.png" alt="実験結果" title="実験結果" width="80%" height="80%"></div>
<div align="center">図1　英語のメールを一年間隔で追加学習した結果</div>
<br>

上記の図では、縦軸がメールに対する識別率、横軸が学習回数を示している。  
なお、識別率は次のように求める。
 - スパムメールと予測したメールの数：SN
 - 正規のメールと予測したメールの数：HN
 - スパムメールと予測したメールの内、実際にスパムメールだった数：TSN
 - 正規のメールと予測したメールの内、実際に正規のメールだった数：THN
このとき、識別率を次の式で求める。

<div align="center"><img src="https://latex.codecogs.com/gif.latex?\frac{TSN&plus;THN}{SN&plus;HN}"></div>

上記の図では、2005年と2006年のメールを学習済みの状態から、2007年のメールデータを追加学習させた場合の識別率の推移を示している。  
図の結果から、既存手法(SGDを用いた場合、上記の図で左のグラフ)の場合、2007年のメールデータを追加学習させると2005年と2006年のメールに対する識別率が低下していることが分かる。つまり、破滅的忘却が発生していることが確認できる。  
これに対して、提案手法(EWCを用いた場合、上記の図で右のグラフ)の場合、2007年のメールデータを追加学習させても、2005年と2007年のメールに対する識別率の低下を軽減できてることが分かる。つまり、破滅的忘却を軽減できていることが確認できる。

次に、日本語と英語のメールを交互に学習させた場合の結果を示す。

**実験条件**  
 - 学習させる順番：日本語→英語と英語→日本語の2パターン
 - 各言語の迷惑メールの内訳：2000通(半分を学習用、残り半分をテスト用として使用)
 - 各言語の正規のメールの内訳：2000通(半分を学習用、残り半分をテスト用として使用)
 - バッチ数:80通(一度に学習するメールの数)
 - Doc2vecによる次元数：300
<br>
<br>
<br>

<div align="center"><img src="https://user-images.githubusercontent.com/26127488/73718446-ddfc8300-475f-11ea-9bb4-673ad3e842b1.jpg"></div>
<div align="center">図2　日本語のメールを学習後に英語のメールを追加学習した結果</div>
<br>
<br>
<br>

<div align="center"><img src="https://user-images.githubusercontent.com/26127488/73718451-e3f26400-475f-11ea-9c68-d6dbbef1d7c7.jpg"></div>
<div align="center">図3　英語のメールを学習後に日本語のメールを追加学習した結果</div>
<br>

図2、図3の見方は図1と同じです。  
図2が日本語のメールを学習した後に、英語のメールを追加学習させた場合の実験結果です。
図2の左の図がEWCを用いなかった場合、図2の右の図がEWCを用いた場合です。
図3のが英語のメールを学習した後に、日本語のメールを追加学習させた場合の実験結果です。
こちらも、図3の左の図がEWCを用いなかった場合、図3の右の図がEWCを用いた場合です。
図2と図3の結果から、EWCを用いなかった場合よりもEWCを用いた場合の方が、初めに学習した言語のメールに対する識別率の低下を軽減できていることがわかります。  
なお、英語と日本語のメールを交互に学習させたのは、意図的に破滅的忘却を発生させるためです。
実際には、メール受信者が複数の言語のメールを交互に受信することは現実的ではありません。

# 補足
2007年に流行したメールに対しての識別率を見ると、EWCを用いた場合よりも既存手法の方が識別率が高くなっていることがわかる。これはEWCを用いた場合の副作用です。EWCを用いない場合では、ニューラルネットワークは新たに学習する2007年のメールデータに対する識別率のみを向上するように学習を行います。これに対し、EWCを用いた場合では、2007年のメールデータに対していの識別率を向上させるだけではなく、過去に学習した2005年と2006年のメールデータに対する識別率の低下を軽減するように学習を行います。その結果、EWCを用いない場合よりもEWCを用いた場合の方が、2007年のメールデータに対する識別率が低くなります。しかし、過去に学習した2005年と2006年のメールデータに対する識別率は、EWCを用いなかった場合よりもEWCを用いた場合の方が高くなっています。そのため、2005年や2006年に流行していたスパムメールが再流行した場合、EWCを用いない既存手法よりも、EWCを用いた提案手法の方が再学習を行うことなく、過去に流行したメールに対して高い識別率を出すことができます。

# 各フォルダの説明
・  
|  
|_01-IMAP:メールサーバからメールを受信するためのプログラム  
|_02-NaiveBayes:ナイーブベイズを用いたスパムフィルタのプログラム  
|_03-SpamFilter_by_NeuralNetwork_using_EWC：私の研究に関するプログラム  
&nbsp;&nbsp; |_doc2vec:doc2vecで生成したベクトルとそのベクトルをt-SNEを用いて二次元の画像にプロットした画像  
&nbsp;&nbsp; |_EWC:EWCを用いたデフォルトのスパムフィルタのプログラム  


# 論文 URL
[論文説明](https://www.ieice.org/ken/paper/20190723N1Of/)

# 学会　受賞歴
[SITE学術奨励賞　2019年7月　「破滅的忘却を軽減するニューラルネットワークを用いたスパムフィルタの提案」](https://www.ieice.org/~site/site_award.html)