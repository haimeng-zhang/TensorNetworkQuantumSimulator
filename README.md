# TensorNetworkQuantumSimulator

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoeyT1994.github.io/TensorNetworkQuantumSimulator.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoeyT1994.github.io/TensorNetworkQuantumSimulator.jl/dev/)
[![Build Status](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoeyT1994/TensorNetworkQuantumSimulator.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoeyT1994/TensorNetworkQuantumSimulator.jl)

A package for simulating quantum circuits and quantum dynamics with tensor networks (TNs) of near-arbitrary geometry. This package is built on top of [ITensors](https://github.com/ITensor/ITensors.jl) and [NamedGraphs](https://github.com/ITensor/NamedGraphs.jl).

The main workhorses of the simulation are _belief propagation_ (BP) and the _Singular Value Decomposition_ for applying gates, and _BP_ or _boundary MPS_ for estimating expectation values and sampling. 

The starting point of most calculations is that you will define a `NamedGraph` object `g` that encodes the geometry of your problem and how you want your tensor network structured. Most simply, the vertices of this graph will correspond to the qubits in your setup and the edges the pairs of qubits that directly interact in your system (e.g. those that will have two-site gates applied to them). Then you define the _circuit_ which you will apply to your tensor network. A circuit, or layer of a circuit, simply takes the form `Vector{<:Tuple}` or `Vector{<:ITensor}` of a list of one or two-site gates to be applied in sequential order. These gates can either be specified as a Tuple `(Gate_string::String, vertices_gate_acts_on::Vector, optional_gate_parameter::Number)` using the pre-defined gates available or as specific ITensors for more custom gate options (see below).    

You can then initialise an `TensorNetworkState ψ` as your chosen starting state,  with user-friendly constructors for various different product states, and apply the desired gates to the TN with the `apply_gates` function. Keyword arguments, i.e. the `apply_kwargs` should be passed to indicate the desired level of truncation to use when applying the circuits. At any point during the simulation, expectation values, samples or other information (e.g. overlaps) can be extracted from the `TensorNetworkState` with different options for the algorithm to use and the various hyperparameters that control their complexity. The relevant literature describes these in more detail. We encourage users to read the literature listed below and look through the examples in the folder [here](examples/) to learn how the code works in more detail so they can effectively deploy their own simulations.

## Upcoming Features
- Applying gates to distant nodes of the TN via SWAP gates.

## Supported Gates
Gates can take the form of ITensors or Tuples of length two or three, i.e.
`(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge})` or `(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge}, optional_parameter::Number)` depending on whether the gate type supports an optional parameter. The qubits_to_act on can be a vector of one or two vertices of the network where the gate acts. In the case of a two-qubit gate an edge of the network can also be passed. 

Pre-Defined One qubit gates (brackets indicates the optional rotation parameter which must be specified). These are consistent with the qiskit definitions.
- "X", "Y", "Z', "Rx" (θ), "Ry" (θ), "Rz" (θ), "CRx" (θ), "CRy" (θ), "CRz" (θ), "P", "H"

Pre-Defined Two qubit gates (brackets indicates the optional rotation angle parameter which must be specified). These are consistent with the qiskit definitions.
- "CNOT", "CX", "CY", "SWAP", "iSWAP", "√SWAP", "√iSWAP", "Rxx" (θ), "Ryy" (θ), "Rzz" (θ), "Rxxyy" (θ), "Rxxyyzz" (θ), "CPHASE" (θ)

If the user wants to instead define a custom gate, they can do so by creating the corresponding `ITensor` which acts on the physical indices for the qubit or pair of qubits they wish it to apply to.

## GPU Support
GPU support is enabled for all operations. Simply load in the relevant GPU Julia module (Metal.jl or CUDA.jl for example) and transform the tensor network, beliefpropagationcache or boundarympscache with `ψ = CUDA.cu(ψ)` and then perform the desired operation e.g. `sample_directly_certified(ψ. 10; norm_message_rank = 10)` and it will run on GPU. Dramatic speedups are seen on NVidia GPUs for moderate to large bond dimension states. 

## Relevant Literature
- [Simulating and sampling quantum circuits with 2D tensor networks](https://arxiv.org/abs/2507.11424)
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)
- [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- [Loop Series Expansions for Tensor Networks](https://arxiv.org/abs/2409.03108)
- [Dynamics of disordered quantum systems with two- and three-dimensional tensor networks](https://arxiv.org/abs/2503.05693)

## Authors
The package was developed by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), an Associate Research Scientist at the Center for Computational Quantum Physics, Flatiron Institute NYC and Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

