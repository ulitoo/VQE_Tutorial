#!/usr/bin/env python
#
# Copyright 2019 the original author or authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This Code is a modification of the original VQE Playground found in:
# https://github.com/JavaFXpert/vqe-playground/
#

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
from qiskit.optimization.applications.ising import max_cut
from qiskit.aqua.operators.legacy import op_converter as op_c

#################################################################################################
# Initializes Optimizacion variables, 1st CIRCUIT , and Expectation grid (hamiltonian based on adjacency matrix)
#################################################################################################
NUM_QUBITS = 5
X = 1 ; Y = 2 ; Z = 3           # Gate Definitions    
circuit_grid_model = None
expectation_grid = None
optimized_rotations = None
cur_rotation_num = 0
min_distance = 0
basis_state_str = ''
proposed_cur_ang_rad = 0
cur_ang_rad = 0
rot_direction = 1
move_radians = np.pi / 8        # Rotation Step

class CircuitGridNode():
    """Represents a node in the circuit grid"""
    ###################################################################
    # Structure of a node is a Gate with properties, 
    # ctrl_a points at the controlling wire of that Gate, -1 is no control
    ###################################################################
    def __init__(self, node_type, radians=0.0, ctrl_a=-1):
        self.node_type = node_type
        self.radians = radians
        self.ctrl_a = ctrl_a
        self.wire_num = -1
        self.column_num = -1

class CircuitGridModel():
    """ Grid-based model of circuit, 2D ARRAY of nodes """

    def __init__(self, max_wires, max_columns):
        ###################################################################
        # Create a 5x21 array of Y , CX , Y , CX , Y , CX , Y ,  CX , Y 
        ###################################################################
        self.max_wires = max_wires
        self.max_columns = max_columns
        self.nodes = np.empty((max_wires, max_columns),
                                dtype = CircuitGridNode)
        for column_num in range(self.max_columns):
            for wire_num in range(self.max_wires):
                if column_num % 5 == 0:
                    self.nodes[wire_num][column_num] = CircuitGridNode(Y, np.pi)
                    self.nodes[wire_num][column_num].wire_num = wire_num
                    self.nodes[wire_num][column_num].column_num = column_num
                else: 
                    if wire_num != 0 and wire_num == column_num % 5:
                        self.nodes[wire_num][column_num] = CircuitGridNode(X, 0, wire_num-1)
                        self.nodes[wire_num][column_num].wire_num = wire_num
                        self.nodes[wire_num][column_num].column_num = column_num

    def get_rotation_gate_nodes(self):
        ###################################################################
        # get a list of nodes suitable for rotation (not controlled)
        # Rotar si es X, Y o Z
        # Don't allow rotation of CONTROLLED  X or Y gates
        # ctrl_a == -1 means que NO es CONTROLLED
        ###################################################################
        rot_gate_nodes = []
        for column_num in range(self.max_columns):
            for wire_num in range(self.max_wires):
                node = self.nodes[wire_num][column_num]
                if node and node.node_type == Y:
                    rot_gate_nodes.append(node)
        return rot_gate_nodes

    def compute_circuit_simple(self):
        ###################################################################
        # build the quantum circuit with quantum gates based on the model.
        # Transform the node 5x21 array into a QISKIT Circuit and return it
        # Use only control X and Rot Y
        ###################################################################

        qr = QuantumRegister(self.max_wires, 'qubit')
        qc = QuantumCircuit(qr)

        for column_num in range(self.max_columns):
            for wire_num in range(self.max_wires):
                node = self.nodes[wire_num][column_num]
                if node:
                    if node.node_type == X: qc.cx(qr[node.ctrl_a], qr[wire_num])
                        # Controlled X gate
                        #qc.cx(qr[node.ctrl_a], qr[wire_num])
                    elif node.node_type == Y: qc.ry(node.radians, qr[wire_num])
                        # Rotation around Y axis
                        #qc.ry(node.radians, qr[wire_num])
        return qc

class ExpectationGrid():
        #########################################################################################
        #  This "grid" contains basis states and Hamiltonian eigenvalues
        #  and calculates expectation value based on it
        #########################################################################################
    def __init__(self, circuit, adj_matrix):
        self.eigenvalues = None
        self.maxcut_shift = 0
        self.basis_states = []
        self.quantum_state = None
        #########################################################################################
        #  basis_states es una array con los nombres de las bases eg |10010>       
        #########################################################################################
        for idx in range(2**NUM_QUBITS):
            state = format(idx, '0' + str(NUM_QUBITS) + 'b')
            self.basis_states.append(state)

        self.set_adj_matrix(adj_matrix)
        self.set_circuit(circuit)

    def set_circuit(self, circuit):
        backend_sv_sim = BasicAer.get_backend('statevector_simulator')
        job_sim = execute(circuit, backend_sv_sim)
        result_sim = job_sim.result()
        self.quantum_state = result_sim.get_statevector(circuit, decimals=3)

    def set_adj_matrix(self, adj_matrix):
        #########################################################################################
        # THIS OUTPUTS Eigenvalues of the Hamiltonian.
        # La funcion Max cut crea un operador Hermitian (Hamiltonian) basado en el entry de un adjacency matrix
        # Should expand for other optimization algorithm
        # La clave es como pasar del problema de optimizacion al Hamiltonian (operator)
        # este es built-in de Qiskit pero estaria bien desarrollar uno ground-up
        #########################################################################################
        maxcut_op, self.maxcut_shift = max_cut.get_operator(adj_matrix)
        maxcut_to_matrix = op_c.to_matrix_operator(maxcut_op)
        self.eigenvalues = maxcut_to_matrix.dia_matrix
    
    def imprime_state(self, statevector_probs):
        state_str = ' Ansatz Quantum State: '
        for idx in range(len(statevector_probs)):
            if statevector_probs[idx] != 0:
                state_str += ("{:.4f}".format(statevector_probs[idx]) + "*|" + self.basis_states[idx] + "> + ")
        print (state_str[:-2])
    
    def calc_expectation_value(self):
        #########################################################################################
        # CALCULO DEL EXPECTATION VALUE del current Quantum State vs HAMILTONIAN
        # <Phi|H|Phi> = sum (eigen*|phi_i|^2) "Energia de ese State"
        #########################################################################################
        statevector_probs = np.absolute(self.quantum_state) ** 2      
        exp_val = np.sum(self.eigenvalues * statevector_probs)

        #########################################################################################
        # Indice del estado que mayor probabilidad tiene. Si hay mas de uno, devuelve solo el 1o
        #########################################################################################
        basis_state_idx = np.argmax(statevector_probs)
        self.imprime_state(statevector_probs)
        #########################################################################################
        # Devuelve el valor esperado del acual estado cuantico y
        # el indice de la componente ortogonal con mayor probabilidad.
        #########################################################################################
        return exp_val, self.basis_states[basis_state_idx]

def calc_new_energy(circuit_grid_model, expectation_grid, rotation_gate_nodes):
    #########################################################################################
    # - This function will CALL QUANTUM CIRCUIT (calc.expect.value)
    #   -set the optimized angles to the "rotable gates" in circuit
    #   -return the cost/distance/energy of the current state (in expectation grid) in Hamiltonian  
    #########################################################################################
    for idx in range(len(rotation_gate_nodes)):
        rotation_gate_nodes[idx].radians = optimized_rotations[idx]

    # Calculate resulting state with new circuit (optimized angles) 
    expectation_grid.set_circuit(circuit_grid_model.compute_circuit_simple())
    # Calculate Energy with new State
    distance, basis_state1 = expectation_grid.calc_expectation_value()
    return distance, basis_state1

def optimize_rotations(circuit_grid_model, expectation_grid, rotation_gate_nodes):
    global min_distance, move_radians, cur_rotation_num, rot_direction, basis_state_str, Fin_optimizacion

    # remember: optimized rotations are the rotations angle for each "rotatable" gate
    # cur_rotation_num es el Gate actual siendo optimizado
    
    if cur_rotation_num < len(optimized_rotations):   # Comprueba que no hemos llegado al ultimo gate optimizable
        
        cur_ang_rad = optimized_rotations[cur_rotation_num]
        proposed_cur_ang_rad = cur_ang_rad
        proposed_cur_ang_rad += move_radians * rot_direction
        
        if (0.0 <= proposed_cur_ang_rad <= np.pi * 2 + 0.01) and (num_times_rot_dir_change[cur_rotation_num]<2):
            
            optimized_rotations[cur_rotation_num] = proposed_cur_ang_rad   
            
            # Calcula nueva energia con nuevo ansatz 
            # CALL QUANTUM CIRCUIT HERE!!!!!!
            temp_distance, basis_state_str = calc_new_energy(circuit_grid_model, expectation_grid, rotation_gate_nodes)
            
            if temp_distance > min_distance: 
                # NOT OPTIMIZED. Distance is increasing so restore the angle in the array and change direction of rotation. + Increase Counter of drection change times
                optimized_rotations[cur_rotation_num] = cur_ang_rad
                rot_direction *= -1
                num_times_rot_dir_change[cur_rotation_num] += 1
            
            else:
                # OPTIMIZED or equal. Distance decreasing, so keep the proposed angle and update Energy
                min_distance = temp_distance

        else:                                       
            cur_rotation_num += 1              # Se salio del rango de 0 a 2pi. o cambio 2 veces de direccion de rotacion.
                                                    # Termina la rotacion este gate AND Move to Next GATE para Rotacion
    else:
        Fin_optimizacion = True    # hemos llegado al ultimo gate optimizable, se acabo el proceso de optimizacion
    
    return min_distance, basis_state_str

#################################################################################################
# Matriz de adyacencia
#################################################################################################
initial_adj_matrix = np.array([
    [0, 3, 1, 3, 0],
    [3, 0, 0, 0, 2],
    [1, 0, 0, 3, 0],
    [3, 0, 3, 0, 2],
    [0, 2, 0, 2, 0]
])

#################################################################################################
# MONTAR EL CIRCUITO INICIAL que pasa de |00000> al estado cuantico ansatz
#################################################################################################
circuit_grid_model = CircuitGridModel(NUM_QUBITS, 21)
circuit = circuit_grid_model.compute_circuit_simple()
expectation_grid = ExpectationGrid(circuit, initial_adj_matrix)

#################################################################################################        
# PRINT Muestra los circuitos y la matriz de adyacencia
#################################################################################################
print("\n#################################################################################################\n")
print("Circuit: \n")
print(circuit)
print("\nAdjacency Matrix is:\n")
print(initial_adj_matrix)
print("\n#################################################################################################\n")

# All optimized rotations intialized with rotation=Pi for all gates at the beginning
# Citcuit and Gates initially are indeed Pauli Y(PI) = Y
rotation_gate_nodes = circuit_grid_model.get_rotation_gate_nodes()
optimized_rotations = np.full(len(rotation_gate_nodes), np.pi)
num_times_rot_dir_change = np.zeros(len(optimized_rotations))

# 1st Optimization Iteration 
min_distance, basis_state_str = calc_new_energy(circuit_grid_model, expectation_grid, rotation_gate_nodes)
    
#################################################################################################
#  -  Main Loop
#################################################################################################
iteration=0
Fin_optimizacion = False        
while not Fin_optimizacion:
    #################################################################################################
    #  -  PRINTOUT de La optimizacion actualmente
    #################################################################################################
    if cur_rotation_num != len(optimized_rotations):
        print(' Iteration :' + str(iteration+1) , ', Rotation of Gate #' , str(cur_rotation_num+1) , 
        'Located in Wire, Column =', rotation_gate_nodes[cur_rotation_num].wire_num+1, ',', 
        rotation_gate_nodes[cur_rotation_num].column_num+1, '. Energy/Cost:', "{:.10f}".format(np.real(min_distance)), 
        ', Maximum Basis State: ', basis_state_str, "\n")
        
    # Optimize
    min_distance, basis_state_str = optimize_rotations(circuit_grid_model, expectation_grid, rotation_gate_nodes)
    iteration +=  1

# FIN and EXIT
print("\n################################################################################################# \n")
print(' min_distance: ', "{:.10f}".format(np.real(min_distance)), ', Optimal Basis State:', basis_state_str)
print("\n################################################################################################# \n")
