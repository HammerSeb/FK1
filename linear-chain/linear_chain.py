#!/usr/bin/env python3
"""
Phonon animation for a 1D diatomic chain with two spring constants (K1, K2).

- Uses pyqtgraph for fast animation.
- Left plot: real-space motion of both branches (acoustic at y=0, optical at y=1).
- Right plot: dispersion relation ω(k) with markers at the currently selected k.
- k can be changed via a slider (0 ... π/a).

Requirements:
    pip install pyqtgraph PyQt5

THIS CODE IS WRITTEN BY CHATGPT 5.1 WITH THE PROMPT:
"Can you write me a python script for the animation of the phonon system of a linear chain with two different spring constants? 
the animation should have a scalable k-vector, show the atom movement on the chain for both branches, and the dispersion relation with an indicator at which k and frequency value we are. I suggest using pyqtgraph but i am open for a different approach"

IT HAS ONLY BEEN SLIGHTLY ADAPTED FOR BETTER VISUALIZATION OF THE CHAINS.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


class PhononChainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Physical parameters ------------------------------------------------
        self.a = 1.0          # lattice constant
        self.m = 1.0          # mass (same for both atoms)
        self.K1 = 1.0         # spring constant within the unit cell
        self.K2 = 0.4         # spring constant between unit cells

        self.n_cells = 8     # number of unit cells in the chain (visualization)
        self.n_k = 400        # k-grid for dispersion and slider
        self.max_amplitude = 0.15 * self.a  # visual scale for displacements

        # Time for animation
        self.t = 0.0
        self.dt = 0.05

        # --- Precompute dispersion & eigenvectors ------------------------------
        self.k_vals, self.omega, self.evecs = self.compute_dispersion()
        # omega shape: (2, n_k); evecs shape: (2, 2, n_k) (basis A/B, branch, k)

        self.current_k_index = self.n_k // 4  # initial k somewhere between 0 and π

        # --- Set up UI ---------------------------------------------------------
        self.init_ui()

        # --- Timer for animation -----------------------------------------------
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)   # in milliseconds

    # -------------------------------------------------------------------------
    # Physics: dispersion of a diatomic chain with equal masses and two K's
    # -------------------------------------------------------------------------
    def compute_dispersion(self):
        """
        Compute dispersion ω(k) and eigenvectors for a diatomic chain
        with equal masses and two spring constants K1, K2.

        We use a basis (u_A, u_B) for each unit cell.

        Dynamical matrix D(k) = (1/m) * [[ K1+K2,        -(K1 + K2 e^{-ika}) ],
                                         [ -(K1 + K2 e^{ika}),  K1+K2       ]]

        Eigenvalues λ = ω^2, eigenvectors give relative A/B amplitudes.
        """
        ks = np.linspace(0.0, np.pi / self.a, self.n_k)
        omega = np.zeros((2, self.n_k), dtype=float)
        evecs = np.zeros((2, 2, self.n_k), dtype=complex)

        for i, k in enumerate(ks):
            diag = (self.K1 + self.K2) / self.m
            off = -(self.K1 + self.K2 * np.exp(-1j * k * self.a)) / self.m

            D = np.array([[diag,        off],
                          [np.conjugate(off), diag]], dtype=complex)

            vals, vecs = np.linalg.eigh(D)  # vals: ω^2, vecs: cols are eigenvectors
            vals = np.real(vals)
            order = np.argsort(vals)        # acoustic first, optical second
            vals = vals[order]
            vecs = vecs[:, order]

            omega[:, i] = np.sqrt(np.clip(vals, 0.0, None))
            evecs[:, :, i] = vecs

        # Normalize eigenvectors so that the maximum absolute component is 1
        # (just for nicer visualization; physics is unchanged).
        max_abs = np.max(np.abs(evecs), axis=0, keepdims=True)
        max_abs[max_abs == 0] = 1.0
        evecs /= max_abs

        return ks, omega, evecs

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------
    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(main_layout)

        # Graphics layout (contains both plots)
        self.graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics)

        # ---------- Real-space plot (left) ----------
        self.real_plot = self.graphics.addPlot(row=0, col=0)
        self.real_plot.setLabel('bottom', 'x / a')
        self.real_plot.setLabel('left', 'branch')
        self.real_plot.setXRange(-0.5, self.n_cells * self.a + 0.5)
        self.real_plot.setYRange(-0.5, 1.5)
        self.real_plot.showGrid(x=True, y=True, alpha=0.3)

        # Label y-axis as 'acoustic' and 'optical'
        axis = self.real_plot.getAxis('left')
        axis.setTicks([[(0, 'acoustic'), (1, 'optical')]])

        # Equilibrium positions of atoms (A and B) along the chain
        self.eq_x_A = np.arange(self.n_cells) * self.a
        self.eq_x_B = self.eq_x_A + 0.5 * self.a

        # Acoustic branch at y=0, optical branch at y=1
        y_acoustic = np.zeros(self.n_cells * 2)
        y_optical = np.ones(self.n_cells * 2)

        x0 = np.concatenate([self.eq_x_A, self.eq_x_B])

        # Scatter plots for acoustic and optical chains
        self.scatter_acoustic = self.real_plot.plot(
            x0, y_acoustic, pen=None, symbol='o', symbolSize=8
        )
        self.scatter_optical = self.real_plot.plot(
            x0, y_optical, pen=None, symbol='o', symbolSize=8, symbolBrush=pg.mkBrush("red")
        )

        # ---------- Dispersion plot (right) ----------
        self.disp_plot = self.graphics.addPlot(row=0, col=1)
        self.disp_plot.setLabel('bottom', 'k a / π')
        self.disp_plot.setLabel('left', 'ω (arb. units)')
        self.disp_plot.showGrid(x=True, y=True, alpha=0.3)

        # Reduced k-axis: ka/πl
        ka_reduced = self.k_vals * self.a / np.pi

        # Plot both branches
        self.disp_plot.plot(ka_reduced, self.omega[0, :], pen=pg.mkPen((50,50,150),width=2))
        self.disp_plot.plot(ka_reduced, self.omega[1, :], pen=pg.mkPen("red",width=2))

        # Vertical line at the current k
        self.k_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(style=QtCore.Qt.DashLine)
        )
        self.disp_plot.addItem(self.k_line)

        # Markers for acoustic and optical frequencies at current k
        self.marker_acoustic = self.disp_plot.plot(
            [ka_reduced[self.current_k_index]],
            [self.omega[0, self.current_k_index]],
            pen=None, symbol='o'
        )
        self.marker_optical = self.disp_plot.plot(
            [ka_reduced[self.current_k_index]],
            [self.omega[1, self.current_k_index]],
            pen=None, symbol='o', symbolBrush=pg.mkBrush("red")
        )

        # ---------- Slider for k ----------
        slider_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(slider_layout)

        slider_layout.addWidget(QtWidgets.QLabel("k:"))

        self.k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.k_slider.setMinimum(0)
        self.k_slider.setMaximum(self.n_k - 1)
        self.k_slider.setValue(self.current_k_index)
        self.k_slider.valueChanged.connect(self.on_k_changed)
        slider_layout.addWidget(self.k_slider)

        self.k_label = QtWidgets.QLabel()
        slider_layout.addWidget(self.k_label)

        self.update_k_label_and_markers()

    # -------------------------------------------------------------------------
    # Slider handling
    # -------------------------------------------------------------------------
    def on_k_changed(self, value):
        self.current_k_index = int(value)
        self.update_k_label_and_markers()

    def update_k_label_and_markers(self):
        k = self.k_vals[self.current_k_index]
        ka_over_pi = k * self.a / np.pi

        self.k_label.setText(f"k a/π = {ka_over_pi:.3f}")

        # Update vertical line and markers in dispersion plot
        x = ka_over_pi
        self.k_line.setPos(x)

        self.marker_acoustic.setData(
            [x], [self.omega[0, self.current_k_index]]
        )
        self.marker_optical.setData(
            [x], [self.omega[1, self.current_k_index]]
        )

    # -------------------------------------------------------------------------
    # Animation
    # -------------------------------------------------------------------------
    def update_animation(self):
        self.t += self.dt

        idx = self.current_k_index
        k = self.k_vals[idx]

        # Acoustic branch (branch 0)
        omega_ac = self.omega[0, idx]
        e_ac = self.evecs[:, 0, idx]   # (e_A, e_B)

        # Optical branch (branch 1)
        omega_op = self.omega[1, idx]
        e_op = self.evecs[:, 1, idx]

        n = np.arange(self.n_cells)  # cell indices

        # Plane-wave phase factor exp(i(kna - ωt))
        phase_ac = np.exp(1j * (k * self.a * n - omega_ac * self.t))
        phase_op = np.exp(1j * (k * self.a * n - omega_op * self.t))

        # Displacements for A and B atoms in each branch
        uA_ac = np.real(e_ac[0] * phase_ac)
        uB_ac = np.real(e_ac[1] * phase_ac)

        uA_op = np.real(e_op[0] * phase_op)
        uB_op = np.real(e_op[1] * phase_op)

        # Scale for visual clarity
        scale = self.max_amplitude

        # Updated positions (longitudinal motion along x)
        xA_ac = self.eq_x_A + scale * uA_ac
        xB_ac = self.eq_x_B + scale * uB_ac

        xA_op = self.eq_x_A + scale * uA_op
        xB_op = self.eq_x_B + scale * uB_op

        # y-positions for the two branches
        y_ac = np.zeros_like(xA_ac)
        y_op = np.ones_like(xA_op)

        # Update scatter data
        self.scatter_acoustic.setData(
            np.concatenate([xA_ac, xB_ac]),
            np.concatenate([y_ac, y_ac])
        )
        self.scatter_optical.setData(
            np.concatenate([xA_op, xB_op]),
            np.concatenate([y_op, y_op])
        )
        # (Dispersion markers are updated only when k changes, which is fine.)

# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
def main():
    app = pg.mkQApp("Phonon animation: diatomic chain with two spring constants")
    w = PhononChainWidget()
    w.setWindowTitle("Phonon system of a linear chain (two spring constants)")
    w.resize(1200, 600)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
