import numpy as np
import matplotlib.pyplot as plt

#Left-right posture

# Hip, Knee, Ankle, Foot:
height0 = [0.99, 0.99, 0.96, 0.93]

bars = ('H+K+A+F', 'K+A+F', 'A+F', 'F')
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
