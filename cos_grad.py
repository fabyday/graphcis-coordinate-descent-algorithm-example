import numpy as np 
import matplotlib.pyplot as plt 
from cycler import cycler

lin = np.linspace(0, 10, num=30)
eps = 0.1
cos_value = np.cos(lin)
sin = -np.sin(lin)
grad_co = (np.cos(lin+eps)-np.cos(lin))/eps

default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=default_cycler)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)


ax0.plot(lin, cos_value)
ax2.plot(lin, sin)
ax3.plot(lin, grad_co)

plt.show()