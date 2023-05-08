import numpy as np
import matplotlib.pyplot as plt

#Left-right posture
# noise 100:
height = [0.99, 0.99, 0.97, 0.95]

#noise 300
height1 = [0.97, 0.96, 0.96, 0.93]

#noise 500
height2 = [0.95, 0.94, 0.96, 0.92]

#noise 1000
height3 = [0.91, 0.91, 0.92, 0.88]

#noise 1500
height4 = [0.89, 0.89, 0.89, 0.84]

#noise 2000
height5 = [0.86, 0.85, 0.88, 0.82]

bars = ('SVM', 'Ada Boost', 'Neural Network', 'Gaussian NB')
y_pos = np.arange(len(bars))

# Noise

# Create bars
barlist = plt.bar(y_pos, height5)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('b')

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()
