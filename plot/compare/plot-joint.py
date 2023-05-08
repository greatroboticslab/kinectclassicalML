import numpy as np
import matplotlib.pyplot as plt

#Left-right posture

# Hip, Knee, Ankle, Foot:
height0 = [0.97, 0.96, 0.97, 0.88]

# Knee, Ankle, Foot:
height = [0.97, 0.96, 0.97, 0.93]

# Ankle, Foot
height1 = [0.94, 0.93, 0.92, 0.86]

# Foot
height2 = [0.88, 0.89, 0.88, 0.84]


bars = ('SVM', 'Ada Boost', 'Neural Network', 'Gaussian NB')
y_pos = np.arange(len(bars))

# Noise

# Create bars
barlist = plt.bar(y_pos, height0)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('b')

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()
