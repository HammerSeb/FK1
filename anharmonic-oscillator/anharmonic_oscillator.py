import numpy as np 
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt


def ode_system(t, y):
    harmonic_k = 1.0
    anharmonic_k = 0.01
    return [y[1], -harmonic_k*y[0] - anharmonic_k*y[0]**2]


t_max = 500
t_step = 0.0005
y0= [5, 0]   #[displacement, velocity]

ode_solution = solve_ivp(ode_system, [0,t_max], y0, dense_output=True, min_step=t_step, max_step = t_step)


freq = fftfreq(len(ode_solution.t), d=t_step)
amp = np.abs(fft(ode_solution.y[0]))**2


f, ax = plt.subplots(1,2)

ax[0].plot(ode_solution.t/(2*np.pi), ode_solution.y[0])
ax[0].set_xlim(0,10)
ax[0].set_ylim(-5.5, 5.5)
ax[0].set_ylabel("displacement [a.u.]")
ax[0].set_xlabel("time [a.u.]")
ax[0].set_title("time domain")
ax[0].grid()

ax[1].plot(freq[0:int(len(freq)/2)]*2*np.pi, amp[0:int(len(freq)/2)])
ax[1].set_xlim(0,5)
ax[1].set_yscale("log")
ax[1].set_ylim(1e6,3e12)
ax[1].set_ylabel("fft amplitude [a.u.]")
ax[1].set_xlabel("frequency [a.u.]")
ax[1].set_title("frequency domain")
ax[1].grid()

f.tight_layout()
f.savefig("anharmonic.png", dpi=300)


def ode_system(t, y):
    harmonic_k = 1.0
    anharmonic_k = 0.01
    return [y[1], -harmonic_k*y[0]]


t_max = 500
t_step = 0.0005
y0= [5, 0]   #[displacement, velocity]

ode_solution = solve_ivp(ode_system, [0,t_max], y0, dense_output=True, min_step=t_step, max_step = t_step)


freq = fftfreq(len(ode_solution.t), d=t_step)
amp = np.abs(fft(ode_solution.y[0]))**2


f, ax = plt.subplots(1,2)

ax[0].plot(ode_solution.t/(2*np.pi), ode_solution.y[0])
ax[0].set_xlim(0,10)
ax[0].set_ylim(-5.5, 5.5)
ax[0].set_ylabel("displacement [a.u.]")
ax[0].set_xlabel("time [a.u.]")
ax[0].set_title("time domain")
ax[0].grid()


ax[1].plot(freq[0:int(len(freq)/2)]*2*np.pi, amp[0:int(len(freq)/2)])
ax[1].set_xlim(0,5)
ax[1].set_yscale("log")
ax[1].set_ylim(1e6,3e12)
ax[1].set_ylabel("fft amplitude [a.u.]")
ax[1].set_xlabel("frequency [a.u.]")
ax[1].set_title("frequency domain")
ax[1].grid()

f.tight_layout()
f.savefig("harmonic.png", dpi=300)

plt.show()