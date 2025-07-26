import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister
import pyscf
import json

# Build N2 molecule
mol = pyscf.gto.Mole()
mol.build(
    atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
    basis="sto-6g",
    symmetry="Dooh",
)

# Define active space
n_frozen = 2
active_space = range(n_frozen, mol.nao_nr())

# Get molecular data and Hamiltonian
scf = pyscf.scf.RHF(mol).run()
mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
norb, nelec = mol_data.norb, mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian
print(f"norb = {norb}")
print(f"nelec = {nelec}")

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()

# Use 2 ansatz layers
n_reps = 2
# Use interactions implementable on a square lattice
pairs_aa = [(p, p + 1) for p in range(norb - 1)]
# pairs_ab = [(p, p) for p in range(norb)]
pairs_ab = [(3, 3)]
ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab)
)

# Construct circuit
qubits = QuantumRegister(2 * norb)
circuit = QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
circuit.measure_all()

tcircuit = circuit.decompose().decompose()
print(f'gates in circuit: {tcircuit.count_ops()}')

# export the circuit to a json file
# the file contains a list of qubit indices and gate instructions
qubit_indices = {'qubit_indices': [q._index for q in tcircuit.qubits]}
lines = [qubit_indices]
for data in tcircuit.data:
    name = data.operation.name
    qubits = tuple([q._index for q in data.qubits])
    params = tuple(data.operation.params) 
    gate = {
        'name': name,
        'qubits': qubits,
        'params': params,
    }
    lines.append(gate)

with open(f'examples/lucj_n2_{norb}o{nelec[0]}e.json', 'w') as f:
    json.dump(lines, f)

# Samlpe from the cricuit
