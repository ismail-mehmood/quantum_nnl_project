#import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/physica:0.9.5": *

#show: ieee.with(
  title: [Comparing Classical Boson Sampling and Stabilised Quantum Circuits for Interferometry],
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

Measuring phase decoherence using a Ramsey interferometer quantum circuit. 

= Beating Ramsey limits with deterministic qubit control

SNR per shot (Rv): Compare phase contrast (vary amplitude) between protocols.

SNR per √time (Rs): Compare sensitivity improvement when including longer stabilized evolution windows.

Breakdown Time (tb): Quantify how long stabilization lasts before vx coherence collapses.

Robustness: Test sensitivity to miscalibrated T1/T2 (simulate with noise models).


This level of quantum information abstraction has applications in quantum sensing,
quantum communication,

= Application of stablising operators to Galton Board

Implementation Steps

    Start with the standard Galton circuit:

        Initialize ball qubit in center peg.

        Use CSWAP-CNOT-CSWAP controlled by the control qubit.

    Insert stabilization between layers:

        After each layer, extract/estimate Bloch components of control qubit (or infer from expected evolution).

        Compute hy(t)∝γ2⋅vx/vzhy​(t)∝γ2​⋅vx​/vz - use eq in paper 2

        Apply Trotterized slices:
        Ustab≈∏ke−iδt [Δσz/2+hy(tk)σy/2]
        Ustab​≈k∏​e−iδt[Δσz​/2+hy​(tk​)σy​/2]

        which in gates is CLOSE ENOUGH TO small Rz(Δδt)Rz​(Δδt) + Ry(hyδt)Ry​(hy​δt).

    Repeat for each Galton layer:

        CSWAP scattering

        Trotterized stabilization pulses

        Move to next layer.


demonstrate with fewer shots