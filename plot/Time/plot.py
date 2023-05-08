import numpy as np
import matplotlib.pyplot as plt

#Left-right posture

# With GPU
height0 = [.01272, 0.7728, 0.1723, 0.0024,  0.3129]

#Without GPU
#height0 = [.01272, 0.7728, 0.1723, 0.0024,  17.604]




bars = ('SVM', 'AdaBoost', 'Neural Network', 'Gaussian NB', "LSTM(GPU)")
y_pos = np.arange(len(bars))

# Noise

# Create bars
barlist = plt.bar(y_pos, height0)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('cyan')

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()
