from qiskit import qpy
import json
import rustworkx as rx
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

norb = 16
nelec = (10, 10)

with open(
    "n2_6-31g_10e16o/bond_distance-1.20000/tcircuit.qpy", "rb"
) as fd:
    tcircuit = qpy.load(fd)[0]

print(
    f"gates in circuit before transpilation: {tcircuit.count_ops()}"
)

# transpile to run on the hardware
# create heavy-hex connectivity graph
num_qubits = 32 + 4
nodes = [i for i in range(num_qubits)]
edges = (
    [(p, p + 1) for p in range(norb - 1)]
    + [(p, p + 1) for p in range(norb, 2 * norb - 1)]
    + [
        (32, 19),
        (3, 32),
        (33, 23),
        (7, 33),
        (34, 27),
        (34, 11),
        (31, 35),
        (35, 15),
    ]
)


edge_list = [(p[0], p[1], None) for p in edges]
graph = rx.PyDiGraph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edge_list)

cmap = CouplingMap()
cmap.graph = graph
# transpile
# TODO: define a target with xx_plus_yy included in the basis gates
tcircuit = transpile(
    tcircuit,
    basis_gates=["rxx", "ryy", "cp", "p", "x", "rx", "measure"],
    coupling_map=cmap,
    initial_layout=list(range(2 * norb)),
)

# export the circuit to a json file
# the file contains a list of qubit indices and gate instructions
qubit_indices = {
    "qubit_indices": [q._index for q in tcircuit.qubits]
}
lines = [qubit_indices]
for data in tcircuit.data:
    name = data.operation.name
    qubits = tuple([q._index for q in data.qubits])
    params = tuple(data.operation.params)
    gate = {
        "name": name,
        "qubits": qubits,
        "params": params,
    }
    lines.append(gate)

with open(f"examples/lucj_n2_{norb}o{nelec[0]}e.json", "w") as f:
    json.dump(lines, f)
print(
    "lucj circuit saved to "
    + f"examples/lucj_n2_{norb}o{nelec[0]}e.json"
)
