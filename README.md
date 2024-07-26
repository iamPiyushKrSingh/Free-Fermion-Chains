# Free Fermion Chain

This package deals with simulating the dynamics of a free fermion chain. The package is written in Python and uses the `numpy`, `scipy`, and `JAX` libraries (for performance). The package is designed to be easy to use and extensible.

Mainly, the package is designed to simulate the numerics of a free fermion chain discussed in the following paper:

- [Measurement on an Anderson Chain](https://doi.org/10.1103/PhysRevB.107.174203) by Paul Pöpperl, Igor V. Gornyi, and Yuval Gefen.
- [Measurement-Induced Phase Transition for Free Fermions above One Dimension](https://doi.org/10.1103/PhysRevLett.132.110403) by Igor Poboiko, Igor V. Gornyi, and Alexander D. Mirlin.
- [Theory of Free Fermions under Random Projective Measurements](https://doi.org/10.1103/PhysRevX.13.041046) by Igor Poboiko, Paul Pöpperl, Igor V. Gornyi, and Alexander D. Mirlin.

## Installation

Since the package is not yet available on PyPI, you can use the package by cloning the repository and importing the package as a module.

## Hamiltonians

The package provides the following Hamiltonians:

1. Anderson Chain:

   $$H = \sum_{i=1}^{N-1} (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i) + \sum_{i=1}^N \varepsilon_i c_i^\dagger c_i$$

   where $\varepsilon_i$ are random numbers drawn from a Gaussian distribution with mean $0$ and standard deviation $\displaystyle \frac{W}{\sqrt{3}}$.

2. Aubry-André Chain:
   $$H = \sum_{i=1}^{N-1} (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i) + \sum_{i=1}^N \cos(2\pi \beta i + \phi) c_i^\dagger c_i$$
   where $\beta$ is an irrational number and $\phi$ is a phase.

## Future Work

The package is in its early stages and only provides the Hamiltonians and a classical dynamics simulation. Also, to use this in my research, I have added the ability to simulate the system's dynamics under projective measurements (of `number operator`). For the performance, I used the `JAX` library. In the future, I plan to add the following features:

1. Add more Hamiltonians.
2. Add numerics from the discussed papers.
3. Have a better structure for the package.
4. Add more tests.
5. Add more documentation.
6. Improve the performance of the package.

## Special Thanks

I thank [Dr. Sambuddha Sanyal]() for his guidance and support in developing this package. I would also like to thank his PhD student, Soumadip Pakrashi, for helping me understand the physics behind the package.
