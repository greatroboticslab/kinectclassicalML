import matplotlib.pyplot as plt
plt.style.use('ggplot')

x = [' Bend', 'Leg Raise/Lower', 'Sit to Stand', '  Turn',	'Jump']
energy = [0.95, 0.99, 0.99, 0.92, 0.99]
ax = plt.subplot(111)
x_pos = [i for i, _ in enumerate(x)]
ax.bar(x_pos, energy, color='black')
ax.bar(x_pos, energy, color='green')
ax.bar(x_pos, energy, color='red')
ax.set_xlabel("Posture")
ax.set_ylabel("Accuracy")

'''
x_pos = [i for i, _ in enumerate(x)]
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax1.bar(x_pos, energy, color='black')
ax1.set_xlabel("Posture")
ax1.set_ylabel("Accuracy")
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
plt.bar(x_pos, energy, color='black')
plt.xlabel("Posture")
plt.ylabel("Accuracy")
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
plt.bar(x_pos, energy, color='black')
plt.xlabel("Posture")
plt.ylabel("Accuracy")
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
plt.bar(x_pos, energy, color='black')
plt.xlabel("Posture")
plt.ylabel("Accuracy")
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
plt.bar(x_pos, energy, color='black')
plt.xlabel("Posture")
plt.ylabel("Accuracy")
'''





#plt.title("The accuracy of classification with different classifiers (without noises added)")

plt.xticks(x_pos, x)

plt.show()
