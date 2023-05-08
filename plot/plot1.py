import numpy as np
import matplotlib.pyplot as plt

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.1      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals =  [0.95, 0.99, 0.99, 0.92, 0.99]
rects1 = ax.bar(ind, yvals, width, color='r')
zvals = [0.95, 0.98, 0.99, 0.77, 0.96]
rects2 = ax.bar(ind+width, zvals, width, color='g')
kvals = [0.95, 0.99, 0.99, 0.93, 0.99]
rects3 = ax.bar(ind+width*2, kvals, width, color='b')
kvals1 = [0.95, 0.99, 0.99,0.92, 0.99]
rects4 = ax.bar(ind+width*3, kvals1, width, color='k')
kvals2 = [0.95, 0.99, 0.99, 0.92, 0.99]
rects5 = ax.bar(ind+width*4, kvals2, width, color='y')

ax.set_ylabel('Accuracy')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('SVM', 'Gaussian NB', 'Random Forest', 'AdaBoost', 'Neural Network') )
ax.legend( (rects1[0], rects2[0], rects3[0],rects4[0],rects5[0]), ('Bend', 'Leg Raise/Lower', 'Sit to Stand', 'Turn', 'Jump'),loc="lower right" )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
        #        ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

plt.show()
