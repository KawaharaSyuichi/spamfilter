import matplotlib.pylab as plt

x1 = [1, 2, 3]
y1 = [0.6707818930041153, 0.6892217170366408, 0.24203821656050956]
x2 = [1.2, 2.2, 3.2]
y2 = [0.4385869565217391, 0.44080953701136677, 0.43859649122807015]
x3 = [1.4, 2.4, 3.4]
y3 = [0.42995169082125606, 0.4380928607089366, 0.25707547169811323]

label_x = ['all', 'spam', 'ham']
plt.bar(x1, y1, color='b', width=0.2, label='2005 & 2006')
plt.bar(x2, y2, color='r', width=0.2, label='2005 & 2007')
plt.bar(x3, y3, color='g', width=0.2, label='2006 & 2007')

plt.legend(loc='best')

plt.xticks([1.2, 2.2, 3.2], label_x)
plt.title('Jaccard')
plt.show()
