import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import hbar, Boltzmann, Avogadro
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages





def bose_distribution(omega: float, temperature: float) -> float:
    """returns Bose distribution at temperature T for angular oscillator frequency omega

    Parameters
    ----------
    omega : float
        angular frequency
    temperature : float
        system temperature 

    Returns
    -------
    float
    """
    return 1 / (np.exp( hbar * omega / (Boltzmann * temperature)) - 1)


def debye_dos(omega: ArrayLike, omega_debye: float, N:float = Avogadro) -> ArrayLike:
    """_summary_

    Parameters
    ----------
    omega : ArrayLike
        _description_
    omega_debye : float
        _description_
    N : float, optional
        _description_, by default Avogadro

    Returns
    -------
    ArrayLike
        _description_
    """
    
    return N * 9 * omega**2 / omega_debye**3


wD = 1e12

# freq = np.linspace(0,1e16, int(1e8))
freq = np.logspace(-3, 15, int(1e4))
T=5
with PdfPages("bose_debye_states.pdf") as pdf:
    f, ax = plt.subplots(1,2, figsize=(10,6))

    f.suptitle("Density of States and Bose distribution at low temperatures")

    ax[0].set_title("linear scale")
    ax[0].plot(freq, bose_distribution(freq, T)*debye_dos(freq,wD), label="DOOS")
    ax[0].plot(freq, 1e12*bose_distribution(freq, T), label="Bose distribution (scaled $\cdot 10^{12}$)")
    ax[0].plot(freq, debye_dos(freq,wD), label="Debye DOS")
    ax[0].set_xlim(0,1e13)
    ax[0].set_ylim(0,5e12)
    ax[0].set_ylabel("dos/doos/n$_b$ [a.u.]")
    ax[0].set_xlabel("angular frequency [1/s]")
    ax[0].legend()

    ax[1].set_title("linear scale")
    ax[1].plot(freq, bose_distribution(freq, T)*debye_dos(freq,wD), label="DOOS")
    ax[1].plot(freq, bose_distribution(freq, T), label="Bose distribution (scaled $\cdot 10^{12}$)")
    ax[1].plot(freq, debye_dos(freq,wD), label="Debye DOS")
    ax[1].set_xlim(1e-3,1e14)
    ax[1].set_ylim(1e-2,1e15)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("dos/doos/n$_b$ [a.u.]")
    ax[1].set_xlabel("angular frequency [1/s]")
    ax[1].legend()

    f.tight_layout()

    pdf.savefig(f)

plt.show()