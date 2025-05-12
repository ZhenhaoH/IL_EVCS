# Scenario data from PJM

Each day is viewed as a scenario that contains 24 steps.

Each scenario is composed of two columns: price $\rho$ and PV generation $P^{PV}$. The unit of price is $/kWh and the PV generation's unit is kW.

- In a npy file, there is a 3-dim array `data`, and `data[:, :, i]` (a $24\times 2$ matrix) denotes
  the `i`-th scenario. Please use the npy file if you want to access the data in Python.
