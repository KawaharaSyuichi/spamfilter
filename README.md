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


# 論文 URL
[論文説明](https://www.ieice.org/ken/paper/20190723N1Of/)

# 学会　受賞歴
[SITE学術奨励賞　2019年7月　「破滅的忘却を軽減するニューラルネットワークを用いたスパムフィルタの提案」](https://www.ieice.org/~site/site_award.html)