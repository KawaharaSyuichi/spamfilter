# panasonicの技術面接用に使おうと考えていたもの
# グラフよりも表の方がわかりやすいと考えたため不採用

import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
fp = FontProperties(
    fname='/Users/kawahara/downloads/ipaexfont00301/ipaexg.ttf')

x1 = [1, 1.5]
y1 = [81.1, 87.2]
x2 = [1.2, 1.7]
y2 = [94.3, 90.3]

label_x = ['既存手法', '提案手法']
plt.bar(x1, y1, color='r', width=0.2, label='2018')
plt.bar(x2, y2, color='b', width=0.2, label='2019')

plt.legend(loc='best')

plt.xticks([1.1, 1.6], label_x, FontProperties=fp)
plt.ylabel('識別率[%]', fontproperties=fp)
plt.ylim([0, 100])
plt.show()
