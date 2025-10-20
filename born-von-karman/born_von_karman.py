import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

def karman_real(ang: float, k:float, amp: float=0.5):
    """illustration of born-von-karman conditions periodic boundary conditions

    Parameters
    ----------
    ang : float
        polar angle 
    k : float
        wave vector
    amp : float, optional
        effective amplitude, by default 0.5
    """
    return amp * np.cos(ang*k) + 1




angle = np.linspace(0, 2*np.pi, 200)

k0 = 1 # k-value for a unit-circle with circumference L=2pi  

f = plt.figure(figsize=(5,8))
f.suptitle("Born-von-Karman periodic boundary conditions")
gs = GridSpec(3,1, height_ratios=[1,5,15], figure=f)

f.subplots_adjust(
    top=0.95,     # space above top axis
    bottom=0.08,  # space below bottom axis
    hspace=0.7,  # vertical space between subplots
    left=0.12,
    right=0.95
)


ax1 = f.add_subplot(gs[1])
ax1.set_title("linear plot")
[karman_linear] = ax1.plot(angle,karman_real(angle,k0))
ax1.set_xticks([0, 2*np.pi],["'front'", "'back'"])
ax1.set_xlim(0, 2*np.pi)
ax1.set_yticks([],[])
ax1.set_xlabel("length L")



ax2 = f.add_subplot(gs[2], projection="polar")

ax2.set_theta_zero_location("N")
ax2.plot(angle, np.ones_like(angle), lw=3, c="k")
[karman_polar] = ax2.plot(angle, karman_real(angle,k0))

ax2.set_rticks([])
ax2.set_xticks([0],["'front/back'"])
ax2.set_title("polar projection")

# axslider = plt.axes([0.1, 0.9, 0.7, 0.05])
axslider = f.add_subplot(gs[0])
slider = Slider(axslider, "k", valmin=1, valmax=10, valinit=k0, valstep=1)

def update(val):
    k =  slider.val
    data = karman_real(angle, k)
    karman_polar.set_data(angle, data)
    karman_linear.set_data(angle, data)
    
    f.canvas.draw_idle()

slider.on_changed(update)

plt.show()

