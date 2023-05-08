import numpy as np
import matplotlib.pyplot as plt

#Left-right posture

# Hip, Knee, Ankle, Foot:
height0 = [0.99, 0.96, 0.96, 0.93, 0.88, 0.86]
#1.5/100, 4.5/100, 8/100, 15/100, 20/100, and 30/100
bars = ('A', 'B', 'C', 'D', 'E', "F")
y_pos = np.arange(len(bars))

# Noise

# Create bars
barlist = plt.bar(y_pos, height0)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('cyan')
barlist[4].set_color('y')
barlist[5].set_color('#FFFF00')

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()
