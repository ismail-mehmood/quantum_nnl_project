#import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/physica:0.9.5": *

#show: ieee.with(
  title: [Quantum Galton Board Implementation: Summary],
  abstract: [
    The Galton Board, also known as a Bean Machine, is a model of the Plinko Game,
    where a ball is modelled as dropping through a series of pegs, with a given probability of 
    falling left or right at each. We here present a summary of the implementation of a quantum version of the 
    classical Galton Board, inspired heavily by @ogpaper.
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
We shall present a summary of our understanding of the implementation of a quantum Galton Board. We shall outline first the workings of the classical Galton Board to serve as a base of intuition, before then moving to an unbiased quantum version, and finally to a biased quantum version (the Universal Statistical Simulator @ogpaper). 

= Classical Galton Board
A model of the Plinko Game, the classical Galton Board arranges pegs in a triangular form, with balls dropped from the top. At each peg, the ball has a 50/50 chance of falling to the right or left. @board shows a typical representation.

#figure(
  image("board-img.jpg", width: 90%),
  caption: [A representation of a classical Galton Board. The resulting distribution of balls is binomial, with p = 0.5. @imgboard],
) <board>

For $n$ levels of pegs, it is apparent that there are $n + 1$ possible exit columns. Each of these is denoted by $k = {0, ..., n}$ from left to right. If we assign a rightward move a value of $+1$, and a leftward move a value of $-1$, then the final position can be given a value $X$, where, for a given column $k$, $ X = k (1) + (n-k) (-1) = 2k-n $ and the probability of achieving a given score is represented by, $ P(X = x) = binom(n, k) p^k (1-p)^(n-k) $evidently a binomial distribution (with $p = 1/2$ in this case). The De Moivre-Laplace Theorem, itself a special case of the Central Limit Theorem, states that as $n$ increases, this approaches a normal distribution, motivating the results we expect from an unbiased Galton Board.

= Unbiased Quantum Galton Board
A Quantum Galton Board is simply a quantum circuit that implements identical logic to the classical Galton Board, using quantum bits (qubits) instead of classical bits. In the unbiased case, we therefore wish to achieve the same Normal distribution of results as in the classical case. We shall use the framework from @ogpaper to explain the quantum circuit implementation of such a board.

== A single unbiased peg
Following the modular approach of @ogpaper, we first discuss the implementation of a single unbiased quantum peg. We make use of 4 qubits, labelled $q_0, q_1, q_2, q_3$, where $q_0$ is the control qubit, $q_2$ represents the input qubit (i.e. this is where the ball enters), and $q_3, q_1$ represent the output of the ball along left or right trajectories respectively. To represent the ball entering the peg, we initialise all four qubits to $ket(0)$, and then apply an $X$ gate to $q_2$, resulting in a state of $ket(q_3q_2q_1q_0) = ket(0100)$. To then implement the unbiased peg logic, we apply a Hadamard gate to the control qubit, which places it into a balanced superposition, modelling the equal probability of falling left or right. We utilise a controlled-SWAP to swap the states of $q_1$ and $q_2$ depending on the control qubit, before utilising a controlled-NOT gate to enforce a state of $ket(1)$ for the control qubit. A final CSWAP then controls the fall (if any) to the left. We measure $q_1$ to determine a rightwards fall, and $q_3$ to assess a leftwards fall. The final resulting state of the circuit is given by, $ 1/sqrt(2) (ket(1001) + ket(0011)) $ which represents an equal probability of falling left (with $q_3 = 1$) or right (with $q_1 = 1$). The implementation of this circuit in @ogpaper is shown in @unbiased-peg. 

#figure(
  image("unbiased-peg.png", width: 80%),
  caption: [The single-peg circuit described above, from @ogpaper.],
) <unbiased-peg>

== Extending to multiple levels
Thanks to this modular approach, the extension of this to $n$ levels is largely straightforward, with a few extra steps each time. For example, with $n = 2$, we have $3$ pegs, and so we need $n+1=3$ output channels and $n+1 = 3$ ancillae, one of which is the control qubit. After each level, we reset $q_0$ to $ket(0)$ and then reapply the Hadamard gate to $q_0$ to re-implement the balanced superposition. We can then simply repeat our one-peg CSWAP-CNOT-CSWAP structure twice, once for each peg on level $2$. The only difference is that we need to apply a CNOT between the logic of each peg, which ensures the control qubit is in a balanced superposition throughout. Measurements for our three possible outcomes are then taken from $q_5, q_3, q_1$ respectively. It is trivial to see that this process generalises for $n$ levels using the modular structure in @ogpaper, by resetting and re-superposing the control qubit between layers, and utilising the CSWAP-CNOT-CSWAP peg logic once for each peg in the layer. Each peg's logic separated by a CNOT to enforce the equal superposition, which ensures that each peg remains unbiased. For $n=2$, the circuit is shown in @unbiased-3-peg. 

#figure(
  image("unbiased-3-peg.png", width: 100%),
  caption: [The circuit for a three-peg quantum Galton Board, from @ogpaper. This generalises to $n$ levels as described.],
) <unbiased-3-peg>

As @ogpaper confirms, the resulting local simulated distribution of outcomes is indeed a normal distribution as $n$ increases, as expected, though the noise associated with the real quantum hardware simulation should be noted.

= Biased Quantum Galton Board
We now turn to the biased version, which effectively allows each peg to have different probabilities of a left or right fall. This can be used to precipitate any statistical distribution, hence the term 'Universal Statistical Simulator' @ogpaper.

== A single biased peg
The implementation of the biased peg is similar to the unbiased peg, but with a key difference. As opposed to the Hadamard gate, we apply a controlled rotation about the x-axis of the Bloch sphere, represented by, $ R_x (theta) = mat(cos(theta/2), -i sin(theta/2); -i sin(theta/2), cos(theta/2)) $ Using $theta = pi/2$ would thus allow us to recover the Hadamard gate, but in general, this produces a biased fall. This results in the single biased peg circuit in @biased-peg. 

#figure(
  image("biased-peg.png", width: 80%),
  caption: [The circuit for implementing a single biased peg, from @ogpaper.],
) <biased-peg>

As @ogpaper demonstrates, the use of $theta = (2 pi)/3$ produces a final state of, $ (sqrt(3/4) ket(0011) + 1/sqrt(4) ket(1001)) $ which gives a 75% probability of a rightwards fall, with only 25% for a leftward fall. By varying $theta$, we also vary this probability ratio, allowing us to achieve any desired split.

== Extending to multiple levels
We can then extend this approach to multiple levels, achieving fine-grained control over the specific bias for each peg. We still reset the control qubit to $ket(0)$ between layers as before, but now, we need to apply a $ket(0)$ reset and a custom controlled-rotation before each individual peg too, which ensures we achive the desired bias for each peg. We need also to reset the control qubit to $ket(0)$ at the end of each layer, ready for the next layer. Finally, to replace the CNOT correction from before, if a row has $i$ pegs, we need $i-1$ CNOTs to correct outputs at the end of each layer. For $n=3$, we thus arrive at the circuit in @biased-3-peg (from @ogpaper), which has removed the final reset for clarity.

#figure(
  image("biased-3-peg.png", width: 100%),
  caption: [The circuit for a three-peg biased quantum Galton Board, with individual control over the bias of each peg, from @ogpaper. This generalises to $n$ levels as described.],
) <biased-3-peg>

