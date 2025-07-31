#import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/physica:0.9.5": *

#show: ieee.with(
  title: [A Comparison of Boson Sampling and Quantum Circuits in Interferometry],
  abstract: [
    Interferometers are devices which use light, or more generally superimposing waves, 
    using interference to extract information. We propose a method by which quantum 
    information can be obtained and used as a benchmark for quantum circuits,
    as well as modelling non-linear interferometers. We simulate these with traditional
    classical methods, then show quantum circuit variants, finally showing how these
    circuits can be used as benchmarks for qubit coherence. 
  ],
  authors: (
    (
      name: "Hayden Dyke",
      organization: [Imperial College London],
      location: [London, United Kingdom],
      email: "hayden.dyke23@imperial.ac.uk"
    ),
    (
      name: "Ismail Mehmood",
      organization: [Imperial College London],
      location: [London, United Kingdom],
      email: "ismail.mehmood24@imperial.ac.uk"
    ),
  ),
)

= Introduction

= Mach-Zehnder Interferometer
The Mach-Zehnder interferometer determines a relative phase shift variation 
between two collimated beams of light. It also has the property that each light
path is only traversed once. It consists of two beam splitters, and a mirror.

In our classically computed photonic model, an MZI is a 2-mode interferometer. 
We use a matrix representation of the beam splitter [1, i, i, 1]/root(2)

To encode this as a quantum circuit REF HERE, we use Hadamard gates to represent
the 50/50 beam splitters, a single qubit as the initial single photon input, and  
a unitary operation U(phi) to represent some phase shift. 

= Michelson Interferometer

= Ramsey Interferometer and Benchmarking