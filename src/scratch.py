"""File for testing ideas"""

import numpy as np
from matplotlib import animation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from models import Node, simulate_cart_pole, dag_pole_fitness

#
# # List for each value
# x_history = [0]
# dx_history = [0]
# theta_history = [np.pi*0]
# dtheta_history = [0]
#
#
# result = step_cart_pole(x_history[-1], dx_history[-1], theta_history[-1], dtheta_history[-1], 1)
# x_history += list(result[0])
# dx_history += list(result[1])
# theta_history += list(result[2])
# dtheta_history += list(result[3])
#
# result = step_cart_pole(x_history[-1], dx_history[-1], theta_history[-1], dtheta_history[-1], -2)
# x_history += list(result[0])
# dx_history += list(result[1])
# theta_history += list(result[2])
# dtheta_history += list(result[3])
#
# for i in range(10):
#     print(i)
#
#     result = step_cart_pole(x_history[-1], dx_history[-1], theta_history[-1], dtheta_history[-1], 0)
#     x_history += list(result[0])
#     dx_history += list(result[1])
#     theta_history += list(result[2])
#     dtheta_history += list(result[3])


x0 = Node('x0')
x1 = Node('x1')
x2 = Node('x2')
x3 = Node('x3')
node = Node(0)

x_history, dx_history, theta_history, dtheta_history = simulate_cart_pole(node)


f = dag_pole_fitness([node])

print(f)




# --- Plotting Results ---
t = list(range(len(x_history)))
fig, axs = plt.subplots(4, 1, sharex=True)

axs[0].plot(t, x_history)
axs[0].axhline(-2.4, color='red')
axs[0].axhline( 2.4, color='red')
axs[0].set_ylabel('Cart Position (m)')
axs[0].grid(True)

axs[1].plot(t, dx_history)
axs[1].axhline(-1, color='red')
axs[1].axhline( 1, color='red')
axs[1].set_ylabel('Cart Velocity (m/s)')
axs[1].grid(True)

axs[2].plot(t, theta_history * 180/np.pi)
axs[2].axhline(-12, color='red')
axs[2].axhline( 12, color='red')
axs[2].set_ylabel('Pole Angle (deg)')
axs[2].grid(True)

axs[3].plot(t, dtheta_history * 180/np.pi)
axs[3].axhline(-1.5, color='red')
axs[3].axhline( 1.5, color='red')
axs[3].set_ylabel('Pole Angular Velocity (deg/s)')
axs[3].grid(True)

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

quit()

time_step = 0.02
time_space = 1000

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

print(len(x_history))

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.3fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):

    pos_0 = [x_history[i], 0]
    pos_1 = [np.sin(theta_history[i])+x_history[i], np.cos(theta_history[i])]

    # history_x = y[:i]
    # history_y = 0[:i]

    line.set_data([pos_0[0], pos_1[0]], [pos_0[1], pos_1[1]])
    # trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*time_step/time_space))
    # time_text.set_text(str(x_history[i]))
    return line, trace, time_text

ani = animation.FuncAnimation(fig, animate, len(x_history), interval=1000*time_step/time_space, blit=True)
plt.show()

