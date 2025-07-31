import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister
import rustworkx as rx
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile
from qiskit import qpy
import pyscf
import json
import numpy as np
from molecules_catalog.util import load_molecular_data
from pathlib import Path
import os

MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

# molecule_name = "n2"
# basis = "6-31g"
# nelectron, norb = 10, 16
molecule_name = "n2"
basis = "cc-pvdz"
nelectron, norb = 10, 26
bond_distance = 1.2
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
operator_filename = f"{molecule_basename}/bond_distance-{bond_distance:.5f}/operator.npz"

operator = np.load(operator_filename)
diag_coulomb_mats = operator["diag_coulomb_mats"]
orbital_rotations = operator["orbital_rotations"]

if molecule_basename == "fe2s2_30e20o":
    mol_data = load_molecular_data(
        molecule_basename,
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )
else:
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{bond_distance:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )
final_orbital_rotation = None
if mol_data.ccsd_t1 is not None:
    final_orbital_rotation = (
        ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
    )
elif mol_data.ccsd_t2 is None:
    nelec = mol_data.nelec
    norb = mol_data.norb
    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
        mol_data.cisd_vec, norb, nelec[0]
    )
    assert abs(c0) > 1e-8
    t1 = c1 / c0
    final_orbital_rotation = (
        ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)
    )

operator = ffsim.UCJOpSpinBalanced(
    diag_coulomb_mats=diag_coulomb_mats,
    orbital_rotations=orbital_rotations,
    final_orbital_rotation=final_orbital_rotation,
)

# Construct circuit
qubits = QuantumRegister(2 * norb)
circuit = QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, mol_data.nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), qubits)
circuit.measure_all()

tcircuit = circuit.decompose().decompose()
print(f'gates in circuit: {tcircuit.count_ops()}')

circuit_filename = f"{molecule_basename}/bond_distance-{bond_distance:.5f}/tcircuit.qpy"
with open(circuit_filename, "wb") as file:
    qpy.dump(tcircuit, file)