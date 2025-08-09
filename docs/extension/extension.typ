#import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/physica:0.9.5": *

#show: ieee.with(
title: [Comparing Classical Boson Sampling and Stabilised Quantum Circuits for Interferometry],
abstract: [
Interferometers are devices which use superimposing waves, causing
interference to extract information. We propose a method by which quantum
information can be obtained and used as a benchmark for quantum circuits,
as well as modelling non-linear interferometers. We simulate these with traditional
classical methods, then show quantum circuit variants, finally showing how these
circuits can be used as benchmarks for qubit coherence, and used to stabilise
qubits in quantum circuits.
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
bibliography: bibliography("refs.bib"),
)

= Introduction
In order to complete these goals, we proceed with the following: first, simulations
of interferometers of general interest to familiarise ourselves with the technology,
followed by a comparison between unitary sampling and quantum circuit methods, before
using the Ramsey interferometry techniques explored with the method outlined in @ramsey_limit
to stabilise the control qubit in our Galton board.

The interferometers of interest are the Mach-Zehnder interferometer (MZI), Michelson
interferometer (MI) and the Ramsey Interferometer (RI). We select these interferometers,
as conveniently they formulate very similarly, with the main differences being the
type of beam splitter used and the position of detectors. These difference correspond to
minimal differences in programming, which reduces circuit complexity, and allows for
an understanding to be built using simpler mechanisms initially.

= Mach-Zehnder Interferometer
The Mach-Zehnder interferometer determines a relative phase shift variation
between two collimated beams of light. It also has the property that each light
path is only traversed once. It consists of two beam splitters, and a mirror.

In our classically computed photonic model, an MZI is a 2-mode interferometer.
We use a matrix representation of the beam splitter:

$ 1/sqrt(2) mat(1, i; i, 1) $

To encode this as a quantum circuit REF HERE, we use Hadamard gates to represent
the 50/50 beam splitters, a single qubit as the initial single photon input, and
a unitary operation U(phi) to represent some phase shift.

#figure(
  image("An-MZI-and-its-equivalent-quantum-circuit.jpg", width: 90%),
  caption: [A quantum circuit equivalent to an MZI. @mzi_quantum],
) <board>

= Michelson Interferometer
The Michelson interferometer differs from the Mach-Zehnder by incorporating a single beam splitter and two mirrors that reflect the beams back to the splitter. This creates a configuration where the beams traverse the same path twice, enhancing sensitivity to certain phase shifts. The corresponding quantum circuit can be modeled using similar Hadamard gates for the beam splitter and controlled phase gates to represent reflections and phase accumulations.

In classical simulations, the Michelson interferometer's double pass can be captured by squaring the unitary evolution of a single pass. In the quantum circuit framework, this translates to applying the equivalent gates twice with appropriate phase shifts. This design also allows for modeling decoherence effects over multiple traversals, providing a natural testing ground for qubit coherence under repeated operations.

= Ramsey Interferometer and Benchmarking

The Ramsey interferometer is particularly valuable for assessing qubit coherence and dephasing times. By preparing a qubit in a superposition state using a Hadamard gate, allowing it to evolve under a controlled phase shift, and then applying a second Hadamard, the resulting interference pattern provides a direct measure of accumulated phase. Decoherence reduces the visibility of this pattern, enabling quantification of noise effects.

Benchmarking using Ramsey interferometry involves measuring the decay of coherence over time or after application of various noise channels, thus providing a figure of merit for quantum device performance. This benchmarking can be integrated within quantum circuits to dynamically adjust control parameters and optimize gate fidelities.

= Beating Ramsey limits with deterministic qubit control

- this top bit is from existing paper, reference and then replace when sims are done
SNR per shot (Rv): Compare phase contrast (vary amplitude) between protocols.

SNR per √time (Rs): Compare sensitivity improvement when including longer stabilized evolution windows.

Breakdown Time (tb): Quantify how long stabilization lasts before vx coherence collapses.

Deterministic control strategies based on feedback from Ramsey interferometry measurements can significantly improve the stability and coherence time of qubits. By implementing adaptive sequences and correction pulses conditioned on the observed phase drift, one can surpass traditional Ramsey limits. This approach enhances signal-to-noise ratios (SNR), especially when leveraging longer evolution windows stabilized by real-time correction.

HOPEFULLY - RUN THESE Simulations incorporating realistic noise models confirm that such stabilizing protocols remain robust, highlighting their potential for information processing applications.

= Application of stabilising operators to Galton Board
Implementation Steps

    Start with the standard Galton circuit:

        Initialize ball qubit in center peg.

        Use CSWAP-CNOT-CSWAP controlled by the control qubit.

    Insert stabilization between layers:

        After each layer, extract or estimate Bloch components of the control qubit (or infer from expected evolution).

        Compute
        $h y(t)∝γ 2⋅v x v z$
        hy​(t)∝γ2​⋅vz​vx​​

        (see equation in paper 2) [REFERENCE MISSING].

        Apply Trotterized slices as
        Ustab≈∏kexp⁡(−iδt(Δ2σz+hy(tk)2σy))
        Ustab​≈k∏​exp(−iδt(2Δ​σz​+2hy​(tk​)​σy​))

        This unitary can be approximated by the sequence of small rotations:
        Ustab≈∏kRz(Δ δt) Ry(hy(tk) δt)
        Ustab​≈k∏​Rz​(Δδt)Ry​(hy​(tk​)δt)

    Repeat for each Galton layer:

        CSWAP scattering

        Trotterized stabilization pulses

        Move to the next layer.

The above protocol effectively reduces the decoherence of the control qubit by counteracting phase noise through a series of carefully timed pulses. This approach harnesses the quantum Zeno effect by continuously steering the qubit state back to its intended trajectory, thereby extending the coherence time and improving the overall fidelity of the Galton board simulation.

Demonstrations HOPEFULLY with fewer shots indicate that lower sampling rates are sufficient to obtain meaningful stabilization, which is crucial for near-term quantum devices with limited measurement budgets.

= Conclusion
We have presented a framework combining classical simulation and quantum circuit implementation of interferometers to benchmark and stabilize qubit coherence. By leveraging well-known interferometric designs such as the Mach-Zehnder, Michelson, and Ramsey interferometers, we bridge traditional optics and quantum information processing techniques.

Our work demonstrates that quantum circuits implementing stabilized interferometric protocols can serve as powerful benchmarks for quantum devices, particularly in the presence of noise and decoherence. The stabilization techniques applied to the quantum Galton board highlight the potential for enhanced control and error mitigation, paving the way for more robust quantum simulations.

Future work includes extending these methods to multi-qubit interferometric configurations and exploring feedback-based real-time stabilization protocols integrated with quantum error correction.
