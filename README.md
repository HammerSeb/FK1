# FK1

code for the lecture "Introduction to solid state physics" at JMU Würzburg - WS 25/26

Each topic has its own directory with all necessary files. Package dependencies are listed in the subsections below.

## Born-von-Karman

Simulation of periodic boundary conditions in linear and polar coordinates. Run with

```shell
python born_von_karman.py
```

Needs `numpy` and `matplotlib`

## Bose-Debye States

Debye density of states multiplied with Bose distribution

```shell
python bose_debye_states.py
```

Needs `numpy` and `matplotlib`

## Linear-chain

Interactive simulation of diatomic linear chain with same atom mass but two different spring constants.

```shell
python bose_debye_states.py
```

Needs `numpy` and `mpyqtgraph`

DISCLAIMER: This code was created using ChatGPT 5.1 and was only slightly adapted for better visibility

## DFT_QE

This is a basic example of band structure calculation with Quantum Espresso. I have used material from [https://pranabdas.github.io/espresso/](https://pranabdas.github.io/espresso/) to put together this example.

You will need a QuantumEspresso installation of your own to run this code. The easiest way to do this is on a Linux OS like Ubuntu. There might be ways to get it to run on Windows/MacOS but I am not aware of any convenient way.
