# -*- coding: utf-8 -*-
"""
Last Modified on 19 August, 2023
@author: Haiyue Kang (Modified from John. F. Kam)
@Link to John. F. Kam's original codes: https://github.com/jkfids/qiskit-entangle/blob/main/code/graphstate.py
"""

# Standard libraries
import itertools
import copy
import random
from math import pi

# Other installed libraries
import numpy as np
import numpy.linalg as la
import queue as Q
import networkx as nx
from numpy import log2, matrix
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix 

# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute, transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix, Operator, PauliList
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_ibm_runtime.options import Options

# Local modules
from utilities import pauli_n, bit_str_list, run_cal, load_cal, pauli_product
from free_entanglebase import Free_EntangleBase
import mthree

# Two-qubit Pauli basis
basis_list = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                  'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ',
                  'ZI', 'ZX', 'ZY', 'ZZ']

# Pauli-Operator matrices
X=matrix([[0,1],[1,0]])
Y=matrix([[0,-1j],[1j,0]])
Z=matrix([[1,0],[0,-1]])
I=matrix([[1,0],[0,1]])
#Indentifying Pauli-Operator in the language of ordered pair (Left marks X, Right marks Z, both marks Y)
P2M_LOOKUP={(0,0):I,(0,1):Z,(1,0):X,(1,1):Y,}
P2S_LOOKUP={(0,0):'I',(0,1):'Z',(1,0):'X',(1,1):'Y',}

#Base Bell State Circuit 
base_circ = QuantumCircuit(2)
base_circ.h(0)
base_circ.cx(0,1)

BS_list_odd = []

# 4 variants of Bellstate
BS_1 = base_circ.copy()
BS_list_odd.append(BS_1)
BS_2 = base_circ.copy()
BS_2.x(0)
BS_list_odd.append(BS_2)
BS_3 = base_circ.copy()
BS_3.z(0)
BS_list_odd.append(BS_3)
BS_4 = base_circ.copy()
BS_4.x(0)
BS_4.z(0)
BS_list_odd.append(BS_4)
BS_list_even = copy.deepcopy(BS_list_odd)
#Bellstate up to local transformation (this is what we should obtain from qst)
for circuit in BS_list_even:
    circuit.h(0)
#Statevectors of the Bell States variants
States_odd = []
States_even = []
for circuit in BS_list_odd:
    state = Statevector(circuit)
    States_odd.append(state)
for circuit in BS_list_even:
    state = Statevector(circuit)
    States_even.append(state)
    
# ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]

def hamming_weight(n):
    """
    return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)
    """
    
    c = 0
    while n:
        c += POPCOUNT_TABLE16[n & 0xffff]
        n >>= 16
    return c

def find_closest_counts(counts, shots):
    """find closest physical counts without negative count and Normalised to shots
    Same algorithm as in https://api.semanticscholar.org/CorpusID:42672869

    Args:
        counts (dictionary): dictionary of counts, key is the bit-string 
        (or integer of the bit-string, value is the count)
        shots (int): number of shots used to obtain this counts

    Returns:
        dictionary: corrected closest physical counts
    """
    scale_factor = shots/sum(counts.values())
    counts.update((k,v*scale_factor/shots) for k,v in counts.items())
    q = Q.PriorityQueue()
    
    count = 0
    for bit_string, v in counts.items():
        q.put((v, bit_string))
        count += 1
    
    a = 0
    continue_zeroing = True
    while not q.empty():
        item = q.get()
        if continue_zeroing:
            if (count > 0) and ((item[0] + a/count) < 0):
                counts[item[1]] = 0
                a += item[0]
                count -= 1
            else:
                continue_zeroing = False
        if not continue_zeroing:
            counts[item[1]] = item[0] + a/count
    
    counts.update((k, v*shots) for k,v in counts.items())
    return counts

class GraphState(Free_EntangleBase):
    """
    Class to run native-graph state negativity measurement experiments

    """

    def __init__(self, backend, qubits_to_connect):
        """
        Initialize from Free_EntangleBase parent class and additionally obtain
        edge adjacencies and generate the native-graph state preperation
        circuit

        """
        super().__init__(backend, qubits_to_connect)  # Inherent from parent class
        
        self.qubits_to_connect = qubits_to_connect
        self.adj_qubits, self.adj_edges = self.__get_adjs()
        self.circuit = self.gen_graphstate_circuit()

        self.batches = None
        self.group_list = None
        self.qst_circuits = None
        self.name_list = None

        self.reps = None
        self.shots = None
        self.qrem = None
        self.sim = None

        self.M_list = None
        self.qrem_circuits = None
        #gate duration times
        durations = InstructionDurations.from_backend(backend)
        self.tx = durations.get('x', 0)
        self.tz = durations.get('rz', 0)
        #objects used for quantum teleportations
        self.teleportation_basis = None
        self.teleported_BellState_circuits = None
        self.teleported_BellState_circuits_qst = None
        #objects used for Fidelity estimation of randomised stabilizers
        self.generators = None
        self.rand_stablizers = None
        self.random_circuits = None
        
        self.partial_angled_qst_circuits = None

    def __get_adjs(self):
        """
        Get the edge-qubit adjacencies for every physical edge in the device.
        Keys are edges (tuple) and values are adjacent qubits (list) or adjacent edges (list of tuples)
        """
        adj_edges = {}
        adj_qubits = {}
        # Iterate over every edge
        for edge in self.edge_list:
            other_edges = self.edge_list.copy()
            other_edges.remove(edge)
            connected_edges = []
            connected_qubits = []
            # Iterate over all other edges
            for edgej in other_edges:
                if np.any(np.isin(edge, edgej)):
                    connected_edges.append(edgej)
                    for qubit in edgej:
                        if qubit not in edge:
                            connected_qubits.append(qubit)
            adj_edges[edge] = connected_edges
            adj_qubits[edge] = connected_qubits

        return adj_qubits, adj_edges

    def gen_graphstate_circuit(self, return_depths=False, angle = pi):
        """
        Generate a native-graph state circuit over every physical edge. Note that the CZ gates have their maximum depths always = 3
        (see Fidel's thesis Fig 3.2)
        Args:
            return_depths (bool, optional): whether return the actual circuit or circuit depth. Defaults to False.
            angle (float): angle in CZ(theta) when coupling the qubits. Defaults to pi.

        Returns:
            _type_: _description_
        """
        circ = QuantumCircuit(self.device_size)
        unconnected_edges = self.edge_list.copy()
        depths = []
        # Apply Hadamard gates to every qubit
        circ.h(self.qubits_to_connect)
        # Connect every edge with CZ gates
        while unconnected_edges:
            connected_qubits = []  # Qubits already connected in the current time step
            remove = []
            # iterate over each edges and add the corresponding CZ gate to the circuit
            for edge in unconnected_edges:
                if np.any(np.isin(edge, connected_qubits)) == False:
                    if angle == pi:
                        circ.cz(edge[0], edge[1])
                    else:
                        circ.cp(angle, edge[0], edge[1])
                    connected_qubits.extend(edge)
                    # remove the edge if it's already added
                    remove.append(edge)
            # Remove connected edges from unconnected edges list
            depths.append(remove)
            for edge in remove:
                unconnected_edges.remove(edge)
                
        if return_depths is True:
            return depths

        return circ
    
    def gen_delay_circuit(self, t, increment, dynamic_decoupling=False):
        """Generate delay circuits based on the graphstate circuit and delay option

        Args:
            t (float>0): total delay time in ns
            increment (float>0): time length of one X-pulses section (if implemented) to divide the total time
            dynamic_decoupling (bool, optional): Delay option. Defaults to False.
        """
        if dynamic_decoupling == False:
            self.delay_circuit = self.circuit.copy()
            if t > 0:
                self.delay_circuit.barrier()
                self.delay_circuit.delay(t)
                
        elif dynamic_decoupling == 'pdd':
            self.delay_circuit = self.circuit.copy()
            if t > 0:
                self.gen_pdd_circuit(t, increment, pulses = 2)
            
        elif dynamic_decoupling == 'hahn_echo':
            self.delay_circuit = self.circuit.copy()
            if t > 0:
                self.gen_hahn_echo_circuit(t)
                
        elif dynamic_decoupling == 'double_pulse':
            self.delay_circuit = self.circuit.copy()
            if t > 0:
                self.gen_double_pulse_circuit(t)
        
    def gen_pdd_circuit(self, t, increment, pulses = 2):
        """Generate periodic dynamical decoupling with 2 pulses in each increment

        Args:
            t (float>0): total delay time
            increment (float>0): time in one section
            pulses (int, optional): number of pulses in one section. Defaults to 2.
        """
        self.dt = increment
        
        tpulses = int(pulses*t/self.dt) # number of wanted pulses per dt*number of dt can fit into total time t = total pulses number
                                        # NOTE tpulses must be an even number for Graph State
        tdelay = t - tpulses*self.tx # total delay time except X gates excution time

        spacings = self.format_delays(
            [tdelay/tpulses]*(tpulses - 1), unit='dt') #tdelay/tpulses = delay time between X gates
                                                       #--> list of delay time between X gates of length tpulse-1
        
        padding = self.format_delays(0.5*(tdelay - spacings.sum()), unit='dt') # padding gives the delay time on both ends of the circuit
        
        self.delay_circuit = self.circuit.copy()
        self.delay_circuit.barrier()
        self.delay_circuit.delay(padding)
        for t in spacings: #spacing has length of odd pulses (tpulses-1)
            self.delay_circuit.x(range(self.nqubits))
            self.delay_circuit.delay(t)
        self.delay_circuit.x(range(self.nqubits))# this adds 1 more pulse -->total pulses is back to even
        self.delay_circuit.delay(padding)
    
    def gen_double_pulse_circuit(self,t):
        """Generate double X pulse dynamical decoupling circuit

        Args:
            t (float>0): total delay time
        """
        tdelay = t-2*self.tx
        padding = self.format_delays(0.25*tdelay, unit = 'dt')
        spacing = self.format_delays(0.5*tdelay, unit = 'dt')
        
        self.delay_circuit = self.circuit.copy()
        self.delay_circuit.barrier()
        
        self.delay_circuit.delay(padding)
        self.delay_circuit.x(range(self.nqubits))
        self.delay_circuit.delay(spacing)
        self.delay_circuit.x(range(self.nqubits))
        self.delay_circuit.delay(padding)
        
    def gen_hahn_echo_circuit(self,t):
        """Generate single X pulse dynamical decoupling circuit

        Args:
            t (float>0): total delay time
        """
        tdelay = t - self.tx
        padding = self.format_delays(0.5*tdelay, unit = 'dt')
        
        self.delay_circuit = self.circuit.copy()
        
        self.delay_circuit.barrier()
        self.delay_circuit.delay(padding)
        self.delay_circuit.x(range(self.nqubits))
        self.delay_circuit.delay(padding)
        
    #def gen_pdd_circuit(self, t, increment):
    #    '''generate periodic dynamical decoupling with imbedded concatenated pulses'''
    #    self.dt = increment
    #    
    #    reps = int(t/self.dt)
    #    
    #    inner_delay_t = (increment - 2*self.tx - 2*self.tz)/4
    #    inner_padding = self.format_delays((inner_delay_t-2*self.tx-2*self.tz)/4, unit='dt')
    #    
    #    inner_circuit = QuantumCircuit(self.nqubits)
    #    for i in range(2):
    #        inner_circuit.x(range(self.nqubits))
    #        inner_circuit.delay(inner_padding)
    #        inner_circuit.z(range(self.nqubits))
    #        inner_circuit.delay(inner_padding)
        
    #    graphstate_circ = self.circuit.copy()
    #    graphstate_circ.barrier()
    #    for i in range(reps):
    #        for j in range(2):
    #            graphstate_circ.x(range(self.nqubits))
    #            graphstate_circ.compose(inner_circuit, inplace = True)
    #            graphstate_circ.z(range(self.nqubits))
    #            graphstate_circ.compose(inner_circuit, inplace = True)
        
    #    self.delay_circuit = graphstate_circ
        
    def format_delays(self, delays, unit='ns'):
        """format the delays

        Args:
            delays (list): list of delay times
            unit (str, optional): unit of delays input. Defaults to 'ns'.

        Returns:
            numpy list: formatted delay times in the unit of qiskit delay gate qunta
        """
        try:
            # For array of times
            n = len(delays)
        except TypeError:
            n = None

        # Convert delays based on input unit
        dt = self.backend.configuration().dt
        if unit == 'ns':
            scale = 1e-9/dt
        elif unit == 'us':
            scale = 1e-6/dt
        elif unit == 'dt':
            scale = 1 #Note since the circuit.delay(t) actually delays the circuit by t(the number)*(the timestep of the machine)
                      #So the actual time we want to delay must be converted into correct unit and divide back the time factor scaled by dt
                      # i.e. for ns it is t*1e-9/dt

        # Qiskit only accepts multiples of 16*dt
        if n is None:
            # For single time
            delays_new = np.floor(delays*scale/16)*16
        else:
            # For array of times
            delays_new = np.zeros(n)
            for i, t in enumerate(delays):
                delays_new[i] = np.round(t*scale/16)*16 #rescale the delay time between X gates

        return delays_new
    
    def gen_generators(self):
        """Find the generators of the graph state stabilizers

        Returns:
            list: list of lists in the [X,Z]-binary locating language of the pauli-strings
        """
        stablizer_generators_str = []
        #find the stablizers (hence elements of Hamiltonian of GraphState) in pauli strings, order from qubit 0 to qubit n
        for qubit, neighbours in self.connections.items():
            pauli_string = 'I'*self.device_size
            pauli_string = pauli_string[:qubit] + 'X' + pauli_string[qubit+1:]
            for neighbour in neighbours:
                pauli_string = pauli_string[:neighbour] + 'Z' + pauli_string[neighbour+1:]
            
            true_pauli_string = ''
            for qubit in self.qubits_to_connect:
                true_pauli_string += pauli_string[qubit]
            stablizer_generators_str.append(true_pauli_string)
        
        stablizer_generators_list = []
        for pauli_string in stablizer_generators_str:
            X_integer = 0  #left/right in tuple is the integer represent X/Z-paulis position after converting to binary
                           #Note if both X,Z == 1 at same position then it represents a Y
            Z_integer = 0
            for i in range(len(pauli_string)):
                # binary to integer conversion (Largest indice on left and Smallest on right, i.e. 100 = 4)
                if pauli_string[i] == 'X':
                    X_integer += 2**(self.nqubits -1 - i)
                elif pauli_string[i] == 'Z':
                    Z_integer += 2**(self.nqubits -1 - i)
            stablizer_generators_list.append([X_integer, Z_integer])
        self.generators = stablizer_generators_list
        
        return stablizer_generators_list
    
    def gen_random_stablizers(self, measurements = 10000):
        """Generate the stabilizers randomly according to the generators

        Args:
            measurements (int, optional): number of random stabilizers. Defaults to 10000.

        Returns:
            dictionary: dictionary of random stabilizers, keys are the pauli-string (in (X,Z)-locating binary
            language), values are the coefficients
        """
        #labels of the random stabilizers in int, interpreted in binary, if a digit is 0, means not chosing
        # this generator, 1 means chosing, selecting from range of 0 to 2^N-1
        rand_stablizers_labels = random.sample(range(0,2**self.nqubits), measurements)
        stablizers = {}
        # iterate over each label and generate the stabilizer
        for i in range(measurements):
            label = rand_stablizers_labels[i]
            stablizer_paulis = [0, 0]#base case, equivalent to identity
            stablizer_coeff = 1
            for j in range(self.nqubits):
                if (label & (1<<j)) != 0: # check each digit of label in binary whether is 0 or 1
                    x1 = stablizer_paulis[0]
                    z1 = stablizer_paulis[1]
                    x2 = self.generators[j][0]
                    z2 = self.generators[j][1]
                    stablizer_paulis[0] ^= x2 #multiply to obtain the new stabilizer
                    stablizer_paulis[1] ^= z2
                    phase_factor = (1j)**( hamming_weight(x1&(~z1)&(~x2)&z2)#XZ
                                          +hamming_weight(x1&z1&x2&(~z2))#YX
                                          +hamming_weight((~x1)&z1&x2&z2)#ZY
                                          -hamming_weight((~x1)&z1&x2&(~z2))#ZX
                                          -hamming_weight(x1&(~z1)&x2&z2)#XY
                                          -hamming_weight(x1&z1&(~x2)&z2))#YZ
                    stablizer_coeff *= phase_factor
            stablizers[tuple(stablizer_paulis)] = stablizer_coeff
        self.rand_stablizers = stablizers
        return stablizers
    
    def gen_random_circuits(self):
        """generate the corresponding QST circuits bases from self.rand_stabilizers

        Returns:
            qiskit circuit list
        """
        stablizers = self.rand_stablizers.copy()
        graphstate = self.circuit.copy() #commented out when using delay circuits
        #graphstate = self.delay_circuit.copy()
        graphstate.barrier()
        
        random_circuits = {}
        #iterate over each stabilizer and generate QST circuit accordingly
        for operator in stablizers.keys():
            basis = self.operator_to_string(operator)
            circ = graphstate.copy(basis)
            for i in range(len(basis)):
                # change basis to the corresponding pauli-operator
                if basis[i] == 'X':
                    circ.h(self.qubits_to_connect[i])
                elif basis[i] == 'Y':
                    circ.sdg(self.qubits_to_connect[i])
                    circ.h(self.qubits_to_connect[i])
            cr = ClassicalRegister(self.nqubits)
            circ.add_register(cr)
            circ.measure(self.qubits_to_connect, cr)
            
            random_circuits[basis] = circ
            
        self.random_circuits = random_circuits
        return random_circuits
    
    def run_random_circuits(self, reps=1, shots=4096, measurements = 10000, qrem = False, sim=None,
                            execution_mode = 'runtime_sampler'):
        """submit the job to IBM backend or on simualtor, jobs are the QST circuits

        Args:
            reps (int, optional): number of repetitions to run the circuit list. Defaults to 1.
            shots (int, optional): number of shots for each circuit. Defaults to 4096.
            measurements (int, optional): number of random stabilizers. Defaults to 10000.
            qrem (bool, optional): whether execute QREM. Defaults to False.
            sim (_type_, optional): simulator mode, ideal, device and actual machine options avaliable. Defaults to None.
            execution_mode (str, optional): old 'execute', transpiled job or runtime sampler avaliable. Defaults to 'runtime_sampler'.

        Returns:
            _type_: qiskit job with list of circuits
        """
        #initialize the genreators and circuits if not done so
        if self.generators is None:
            self.gen_generators()
        if self.rand_stablizers is None:
            self.gen_random_stablizers(measurements=measurements)
        random_circuits = self.gen_random_circuits()
        #transform circuits from dictionary to a list
        circ_list = []
        for circ in random_circuits.values():
            circ_list.append(circ)
        #extend the list to number of repetitions
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        circ_list = circ_list_multi
        #add QREM circuit if necessary
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)
        #execute the job
        if sim is None:
            #run on actual IBM machine
            if execution_mode == 'execute':
                job = execute(circ_list, backend=self.backend, shots=shots)
            elif execution_mode == 'transpile':
                circ_list_transpiled = transpile(circ_list, backend = self.backend)
                job = self.backend.run(circ_list_transpiled, shots=shots)
            elif execution_mode == 'runtime_sampler':
                service = QiskitRuntimeService()
                backend = service.backend(self.backend.name)
                options = Options()
                options.execution.shots = shots
                options.max_execution_time = 7200
                options.optimization_level = 1 #same as QREM

                with Session(service=service, backend=backend) as session:
                    sampler = Sampler(session=session, options=options)
                    job = sampler.run(circ_list)
            
        elif sim == "ideal":
            #execute on noiseless simulator
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.device_size)),
                          shots=shots)
        elif sim == "device":
            #run with noisemodel
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        return job
    
    def counts_from_random_results(self, result, rand_stablizers, sampler):
        """collect results from job and store into dictionary list

        Args:
            result : IBMJob.result()
            rand_stablizers (dictionary): dictionary of random stabilizers
            sampler (bool): true if the job was executed on runtime sampler

        Returns:
            list: list of dictionarys storing counts of the result
        """
        if self.rand_stablizers is None:
            self.rand_stablizers = rand_stablizers
        #find the reps, shots and whether qrem was executed
        if self.reps is None:
            if sampler is True:
                self.reps = int(len(result.quasi_dists)/len(self.rand_stablizers.keys()))
                self.shots = result.metadata[0]['shots']
                self.qrem = True
            else:
                self.reps = int(len(result.results)/len(self.rand_stablizers.keys()))
                self.shots = result.results[0].shots
                try:  # Try to obtain QREM results
                    result.get_counts('qrem0')
                    self.qrem = True
                except:
                    self.qrem = False
        
        if sampler is True:
            basis_counts_list = []
            x = 0 #counter/index for the circuit when it is executed
            for i in range(self.reps):
                basis_counts = {basis: {} for basis in self.rand_stablizers.keys()}
                for operator in self.rand_stablizers.keys():
                    counts = result.quasi_dists[x]#get the counts from result in the order when it is executed
                    counts_zipped = {}
                    #transform into integer storage instead of bit-string to save memory
                    for integer, prob in counts.items():
                        bit_str = bin(integer)[2:].zfill(self.nqubits)
                        idx = int(bit_str[::-1],2)#transform into bottom-ending notation (top in circuit=left in bit string)
                        counts_zipped[idx] = prob*self.shots
                    basis_counts[operator] = counts_zipped 
                    x += 1
                basis_counts_list.append(basis_counts)
        
            # Save list of calibration matrices for each qubit
            if self.qrem is True:
                qrem_counts = [result.quasi_dists[-2], result.quasi_dists[-1]]
                M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
                for jj, distribution in enumerate(qrem_counts):
                    for integer, prob in distribution.items():
                        bit_str = bin(integer)[2:].zfill(self.device_size)
                        for i, q in enumerate(bit_str[::-1]):
                            ii = int(q)
                            M_list[i][ii, jj] += prob

                self.M_list = M_list
            
        else:
            basis_counts_list = []
            for i in range(self.reps):
                basis_counts = {basis: {} for basis in self.rand_stablizers.keys()}
                for operator in self.rand_stablizers.keys():
                    basis = self.operator_to_string(operator)
                    name_ext = basis + f'-{i}'
                    counts = result.get_counts(name_ext)#obtain result as counts in a dictionary
                    counts_zipped = {}
                    #transform into integer storage instead of bit-string to save memory
                    for bit_string, count in counts.items():
                        idx = int(bit_string[::-1],2)#transform into bottom-ending notation (top in circuit=left in bit string)
                        counts_zipped[idx] = count
                    basis_counts[operator] = counts_zipped 
                
                basis_counts_list.append(basis_counts)

            # Save list of calibration matrices for each qubit
            if self.qrem is True:
                qrem_counts = [result.get_counts('qrem0'),
                               result.get_counts('qrem1')]
                M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
                for jj, counts in enumerate(qrem_counts):
                    for bit_str, count in counts.items():
                        for i, q in enumerate(bit_str[::-1]):
                            ii = int(q)
                            M_list[i][ii, jj] += count
                # Normalise
                norm = 1/self.shots
                for M in M_list:
                    M *= norm
                self.M_list = M_list
        
        return basis_counts_list #, basis_pvecs_list
    
    def apply_qrem_random_stablizers(self, counts_list):
        """Apply QREM to counts_list 

        Args:
            counts_list (list): list of counts obtained from 
            self. counts_from_random_results

        Returns:
            list: return the error mitigated counts_list
        """
        counts_mit_list = []
        for i in range(self.reps):
            counts_mit_dict = {operator: {} for operator in counts_list[i].keys()}
            # Invert n-qubit calibration matrix
            M_inv = la.inv(self.calc_M_multi(self.qubits_to_connect))
            for operator, counts in counts_list[i].items():
                pvec = np.zeros(2**self.nqubits)
                for idx, count in counts.items():
                    pvec[idx] += count
                # "Ideal" probability vector
                pvec_mit = np.matmul(M_inv, pvec)
                for ii, count in enumerate(pvec_mit):
                    counts_mit_dict[operator][ii] = count
                
            counts_mit_list.append(counts_mit_dict)

        return counts_mit_list
        
        
    def apply_reduced_qrem_random_stablizers(self, counts_list, mitigate_qubits = [1,3,4,5,6], threshold = 0.02):
        """Apply QREM qubit-wisely

        Args:
            counts_list (list): list of counts obtained from self. counts_from_random_results
            mitigate_qubits (list, optional): list of qubtis index to mitigate. Defaults to [1,3,4,5,6].
            threshold (float, optional): threshold such that a count is zero-out if below it. Defaults to 0.02.

        Returns:
            _type_: counts_list qubit-wise mitigated
        """
        counts_list_mit = copy.deepcopy(counts_list)
        for i in range(self.reps):
            for operator, counts in counts_list[i].items():
                corrected_counts = copy.deepcopy(counts)
                #iterate over each qubit to mitigate
                for q in mitigate_qubits:
                    idx = self.qubits_to_connect.index(q)
                    calibration_M = la.inv(self.M_list[q])
                    applied_names = set([])
                    corrected_int = [k for k in corrected_counts.keys()]
                    # iterate over each bit-string in the counts, correct its qth element
                    for bit_string_int in corrected_int:
                        bit_string = bin(bit_string_int)[2:].zfill(self.nqubits)
                        bit_string_int_orig = int(bit_string[::-1],2)
                        bit_string_list = list(bit_string)
                        bit_string_list[idx] = '_'
                        name = "".join(bit_string_list)
                        #check if this bit-string has been corrected
                        if name not in applied_names:
                            applied_names.add(name)
                            if (bit_string_int_orig & (1 << idx)) != 0:
                                bit_string_list[idx] = '0'
                            else:
                                bit_string_list[idx] = '1'
                            bit_string_flip = "".join(bit_string_list)
                            bit_string_int_flip = int(bit_string_flip,2)
                            
                            reduced_pvec = np.zeros(2)
                            if bit_string_int < bit_string_int_flip:
                                if bit_string_int in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string_int]
                                if bit_string_int_flip in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string_int_flip]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > threshold:
                                    corrected_counts[bit_string_int] = reduced_pvec_mit[0]
                                else:
                                    corrected_counts[bit_string_int] = 0
                                    del corrected_counts[bit_string_int]
                                if abs(reduced_pvec_mit[1]) > threshold:
                                    corrected_counts[bit_string_int_flip] = reduced_pvec_mit[1]
                                else:
                                    corrected_counts[bit_string_int_flip] = 0
                                    del corrected_counts[bit_string_int_flip]
                                
                            else:
                                if bit_string_int in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string_int]
                                if bit_string_int_flip in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string_int_flip]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > threshold:
                                    corrected_counts[bit_string_int_flip] = reduced_pvec_mit[0]
                                else:
                                    corrected_counts[bit_string_int_flip] = 0
                                    del corrected_counts[bit_string_int_flip]
                                if abs(reduced_pvec_mit[1]) > threshold:
                                    corrected_counts[bit_string_int] = reduced_pvec_mit[1]
                                else:
                                    corrected_counts[bit_string_int] = 0
                                    del corrected_counts[bit_string_int]
                corrected_counts = find_closest_counts(corrected_counts, self.shots)
                print(f'{operator} done, size {len(corrected_counts)}')
                counts_list_mit[i][operator] = corrected_counts
                
            print(f'rep {i} done')
        return counts_list_mit
    
    def expected_vals_from_random_stablizers(self, result, rand_stablizers, mitigate_qubits, apply_mit = 'reduced_QREM',
                                             sampler = False):
        """calculate the expected values from each stabilizer measurement results
        
        Args:
            result (IBMJob.result()): job result
            rand_stablizers (dictionary): random stabilizers
            mitigate_qubits (list): list of qubits to mitigate
            apply_mit (str, optional): mitigation mode. Defaults to 'reduced_QREM'.
            sampler (bool, optional): whether job was executed using runtime sampler. Defaults to False.

        Returns:
            dictionary: dictionary of expected values, keys are the stabilizers and values are the expected values
        """
        #obtain probabilities vectors from each QST basis and each repitition
        counts_list = self.counts_from_random_results(result, rand_stablizers=rand_stablizers,
                                                      sampler = sampler)
        # If mitigation is applied
        if apply_mit == 'reduced_QREM':
            counts_list = self.apply_reduced_qrem_random_stablizers(counts_list, mitigate_qubits=mitigate_qubits)
        elif apply_mit == 'QREM':
            counts_list = self.apply_qrem_random_stablizers(counts_list)
        random_expected_values = {operator: [] for operator in rand_stablizers.keys()}
        for i in range(self.reps):
            #load information of expected values from non-identity basis
            for operator, counts in counts_list[i].items():
                non_identities = operator[0]|operator[1]
                expected_val = 0
                for bit_str_int, count in counts.items():
                    expected_val += count*(-1)**hamming_weight(non_identities&bit_str_int)
                expected_val *= rand_stablizers[operator] #multiply by phase factor
                expected_val /= self.shots
                random_expected_values[operator].append(expected_val)
            
        return random_expected_values
    
    def fidelities_from_random_stablizers(self, expected_vals_dict):
        """calculate the fidelities from the expected values
        same algorithm as in DOI: 10.1103/PhysRevLett.106.230501
        Args:
            expected_vals_dict (dictionary): expected values

        Returns:
            list: fidelities of the graphstate (in each repetition)
        """
        fidelities = np.zeros(self.reps, dtype = complex)
        for expected_vals_list in expected_vals_dict.values():
            fidelities += np.array(expected_vals_list)
        fidelities /= len(expected_vals_dict.keys())
        
        return fidelities
    
    def operator_to_string(self, operator):
        """convert the (X,Z)-locating binary language of pauli-string into real pauli-string

        Args:
            operator (tuple): tuple of (X,Z) operator identifying its positon inthe pauli-string

        Returns:
            str: pauli-string
        """
        x,z = operator
        out = max(x,z)
        if not out:
            return 'I'*self.nqubits
        
        sz=int(log2(float(out)))+1
        #Note Integer reads large to small from L to R in binary, so need to reverse the pauli string
        pauli_string = ''.join(P2S_LOOKUP[(bool(x&2**ix),bool(z&2**ix))] for ix in range(sz)).ljust(self.nqubits,'I')[::-1]
        return pauli_string
    
    def gen_partial_angled_qst_circuits(self, reduced_qubits = None, angle_counts = 10):
        """generate non-maximally entangled graph state circuit and then QST circuits

        Args:
            reduced_qubits (list, optional): list of two qubits in a pair. Defaults to None.
            angle_counts (int, optional): total number of scans from 0 to pi. Defaults to 10.

        Returns:
            dictionary: dictionary of circuits
        """
        angles = np.linspace(0,pi,angle_counts)
        # Generate batches of groups (target edges + adjacent qubits) to perform
        # QST in parallel
        circuits = {}  # Dictionary of groups of circuits where batches are keys
        name_list = []
        #iterate over the angles
        for i in range(len(angles)):
            # Dictionary of circuits where measurement basis are keys
            qst_circuits = {}
            #generate the graphstate but CZ at angle not pi
            graphstate = self.gen_graphstate_circuit(angle=angles[i])
            graphstate.barrier()
            
            if reduced_qubits != None:
                targ_array = np.array(reduced_qubits)
            else:
                targ_array = np.array(self.qubits_to_connect)
            flat_array = targ_array.flatten()
            
            # Create circuits for each basis combination over target pairs
            circxx = graphstate.copy(f'angle{i}-XX')
            circxx.h(flat_array)
            qst_circuits['XX'] = circxx

            circxy = graphstate.copy(f'angle{i}-XY')
            circxy.sdg(targ_array[1].tolist())
            circxy.h(flat_array)
            qst_circuits['XY'] = circxy

            circxz = graphstate.copy(f'angle{i}-XZ')
            circxz.h(targ_array[0].tolist())
            qst_circuits['XZ'] = circxz

            circyx = graphstate.copy(f'angle{i}-YX')
            circyx.sdg(targ_array[0].tolist())
            circyx.h(flat_array)
            qst_circuits['YX'] = circyx

            circyy = graphstate.copy(f'angle{i}-YY')
            circyy.sdg(flat_array)
            circyy.h(flat_array)
            qst_circuits['YY'] = circyy

            circyz = graphstate.copy(f'angle{i}-YZ')
            circyz.sdg(targ_array[0].tolist())
            circyz.h(targ_array[0].tolist())
            qst_circuits['YZ'] = circyz

            circzx = graphstate.copy(f'angle{i}-ZX')
            circzx.h(targ_array[1].tolist())
            qst_circuits['ZX'] = circzx

            circzy = graphstate.copy(f'angle{i}-ZY')
            circzy.sdg(targ_array[1].tolist())
            circzy.h(targ_array[1].tolist())
            qst_circuits['ZY'] = circzy

            circzz = graphstate.copy(f'angle{i}-ZZ')
            qst_circuits['ZZ'] = circzz
            
            for circ in qst_circuits.values():
                name_list.append(circ.name)
                if reduced_qubits != None:
                    cr = ClassicalRegister(self.nqubits)
                    circ.add_register(cr)
                    #measurement order in neighbours + qubit pair
                    order = sorted(list(set(self.qubits_to_connect)-set(reduced_qubits))) + reduced_qubits
                    circ.measure(order, cr)
                else:
                    cr = ClassicalRegister(2)
                    circ.add_register(cr)
                    circ.measure(self.qubits_to_connect, cr)
            circuits[f'angle{i}'] = qst_circuits

        self.partial_angled_qst_circuits = circuits
        self.name_list = name_list

        return circuits
    
    def run_partial_angled_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None, execution_mode='transpile'):
        """run the job of partial angled graph state QST circuits

        Args:
            reps (int, optional): number of repetitions. Defaults to 1.
            shots (int, optional): number of shots of each circuit. Defaults to 4096.
            qrem (bool, optional): whether execute QREM circuits. Defaults to False.
            sim (_type_, optional): simulator mode, (noiseless, noisemodel and real machine). Defaults to None.
            execution_mode (str, optional): execution mode (execute, transpile/run and runtime sampler). Defaults to 'transpile'.

        Returns:
            _type_: _description_
        """
        self.reps = reps
        self.shots = shots
        self.qrem = qrem
        self.sim = sim

        # Convert circuits dict into list form
        circ_list = []
        for angled_circuits in self.partial_angled_qst_circuits.values():
            for circuit in angled_circuits.values():
                circ_list.append(circuit)

        # Extend circuit list by number of repetitions
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        circ_list = circ_list_multi

        # Generate QREM circuits and append to circ_list if qrem == True
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)

        # If circuits are executed on a simulator or real backend
        if sim is None:
            if execution_mode == 'execute':
                job = execute(circ_list, backend=self.backend, shots=shots)
            elif execution_mode == 'transpile':
                circ_list_transpiled = transpile(circ_list, backend = self.backend)
                job = self.backend.run(circ_list_transpiled, shots=shots)
            elif execution_mode == 'runtime_sampler':
                service = QiskitRuntimeService()
                backend = service.backend(self.backend.name)
                options = Options()
                options.execution.shots = shots
                options.max_execution_time = 7200 #check this later
                options.optimization_level = 1

                with Session(service=service, backend=backend) as session:
                    sampler = Sampler(session=session, options=options)
                    job = sampler.run(circ_list)
        # If circuits are executed on noise model (or noiseless simulator)
        elif sim == "ideal":
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.device_size)),
                          shots=shots)
        elif sim == "device":
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)
        return job
    
    def partial_angled_counts_from_result(self, result, angle_counts, reduced_qubits=None):
        """obtain counts from partially entangled graph state job

        Args:
            result (IBMJob.result()): job
            angle_counts (int): number of equally separated angles from 0 to pi
            reduced_qubits (list, optional): list of two qubits in pair. Defaults to None.

        Returns:
            list: list of dictionaries storing counts from result
        """
        #obtain repetitions, shots and whether QREM was executed in the job
        if self.partial_angled_qst_circuits is None:
            self.gen_partial_angled_qst_circuits(reduced_qubits=reduced_qubits, angle_counts = angle_counts)
        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))# number of qubits？/total number of QST circuits need to run 
                                                                    #s.t. all pairs are fulled QST
            self.shots = result.results[0].shots
            try:  # Try to obtain QREM results
                result.get_counts('qrem0')
                self.qrem = True
            except:
                self.qrem = False

        # Load counts as dict experiment-wise
        qst_counts_multi = []
        qst_pvecs_multi = []
        #iterate over each repetition of same circuits
        for i in range(self.reps):
            qst_counts = {f'angle{i}': {} for i in range(angle_counts)}
            qst_pvecs = {f'angle{i}': {} for i in range(angle_counts)}
            for name in self.name_list:
                angle, basis = name.split('-')
                name_ext = name + f'-{i}'
                counts = result.get_counts(name_ext)
                qst_counts[angle][basis] = counts #e.g. qst_counts[pi/2]['ZZ'] = {'000000':21,'000011':12,....}
                pvec = np.zeros(2**self.nqubits)
                for bit_str, count in counts.items():
                    pvec[int(bit_str[::-1],2)] += count
                pvec /= self.shots
                qst_pvecs[angle][basis] = pvec
            qst_counts_multi.append(qst_counts) #e.g. qst_counts_multi = [{qst_counts 1st iter}, {qst_counts 2nd iter},...]
            qst_pvecs_multi.append(qst_pvecs)

        # Save list of calibration matrices for each qubit
        if self.qrem is True:
            qrem_counts = [result.get_counts('qrem0'),
                           result.get_counts('qrem1')]

            M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
            for jj, counts in enumerate(qrem_counts):
                for bit_str, count in counts.items():
                    for i, q in enumerate(bit_str[::-1]):
                        ii = int(q)
                        M_list[i][ii, jj] += count

            # Normalise
            norm = 1/self.shots
            for M in M_list:
                M *= norm

            self.M_list = M_list
        
        return qst_counts_multi, qst_pvecs_multi
        
    def apply_qrem_partial_angled(self, g_counts, g_vecs, reduced_qubits):
        """Apply QREM to angled graph state counts_list

        Args:
            g_counts (list): list of counts from result
            g_vecs (list): list of probability vectors from result
            reduced_qubits (list): list of two qubits pair

        Returns:
            lists: the error mitigated counts_list and pvec_list
        """
        g_counts_list = []
        g_vecs_list = []
        if reduced_qubits != None:
            #order of tensor product same as measurement order
            order = sorted(list(set(self.qubits_to_connect)-set(reduced_qubits))) + reduced_qubits
        else:
            order = self.qubits_to_connect
        M_inv = la.inv(self.calc_M_multi(order))
        for i in range(self.reps):
            g_counts_mit = g_counts[i].copy()
            g_vecs_mit = g_vecs[i].copy()

            for angle, pvecs in g_vecs[i].items():
                for basis, vec in pvecs.items():
                    # "Ideal" probability vector
                    vec_mit = np.matmul(M_inv, vec)
                    g_vecs_mit[angle][basis] = vec_mit
                    # Equivalent ideal group counts
                    for ii, count in enumerate(vec_mit):
                        bit_str = bin(ii)[2:].zfill(self.nqubits)[::-1]
                        g_counts_mit[angle][basis][bit_str] = count*self.shots

            g_counts_list.append(g_counts_mit)
            g_vecs_list.append(g_vecs_mit)

        return g_counts_list, g_vecs_list
    
    def bin_pvecs_partial_angled_counts(self, qst_counts):
        """categorise the probability vectors according to measurement results of neighbours

        Args:
            qst_counts (list): list of counts from result

        Returns:
            list: list of counts categorised into each possible neighbour bit-string
        """
        pvecs_list = []
        
        for i in range(self.reps):
            n = self.nqubits - 2#number of qubits in neighbours of the pair
            pvecs_dict = {bn: {angle: {basis: np.zeros(4) for basis in basis_list} 
                               for angle in qst_counts[i].keys()} 
                          for bn in bit_str_list(n)}
            
            for angle, basis_counts in qst_counts[i].items():
                for basis, counts in basis_counts.items():
                    for bit_str, count in counts.items():
                        idx = int(bit_str[:2][::-1], 2) # qubit pair bit-string
                        bn = bit_str[2:][::-1]# neighbour bit-string
                        pvecs_dict[bn][angle][basis][idx] += count

                    for bn in bit_str_list(n):
                        pvec = pvecs_dict[bn][angle][basis]
                        norm = 1/pvec.sum()
                        pvecs_dict[bn][angle][basis] = pvec*norm
            
            pvecs_list.append(pvecs_dict)

        return pvecs_list
    
    def recon_density_mats_from_partial_angled(self, result, apply_mit=None, angle_counts=10, reduced_qubits=None):
        """Reconstruct the density matrices for every qubit pair for every
        measurement combination and save it as a dictionary or list of
        dictionaries

        Args:
            result (IBMJob.result()): job result
            apply_mit (bool, optional): mitigation mode. Defaults to None.
            angle_counts (int, optional): total number of angles tested. Defaults to 10.
            reduced_qubits (list, optional): list of two qubits in a pair. Defaults to None.

        Returns:
            list: list of dictionaries of density matrices
        """
        rho_dict_list = []
        #obtain counts/probability vectors from result
        qst_counts, qst_pvecs = self.partial_angled_counts_from_result(result, angle_counts=angle_counts,
                                                                       reduced_qubits=reduced_qubits)# obtain raw data in the batch form
        
        # Whether mitigation is applied or not defaults to whether qrem circuits
        # are included in result
        if apply_mit is None:
            apply_mit = self.qrem
        # If mitigation is applied
        if apply_mit is True:
            qst_counts, qst_pvecs = self.apply_qrem_partial_angled(qst_counts, qst_pvecs, reduced_qubits=reduced_qubits)#convert raw counts into error mitigated counts
                                                                #Note M matrices are already calculated in counts_from result function
                                                                #which are then used in apply_qrem    
        if reduced_qubits != None:
            qst_pvecs = self.bin_pvecs_partial_angled_counts(qst_counts)
                                                        
        angles = np.linspace(0,pi,angle_counts)
        #iterate over each repetition of same circuits
        for i in range(self.reps):
            # if choose to include neighbours
            if reduced_qubits != None:
                rho_dict = {bn: {} for bn in bit_str_list(self.nqubits-2)}
                for bn, angle_dict in qst_pvecs[i].items():
                    for angle_i, pvecs in angle_dict.items():
                        angle = angles[int(angle_i[-1])]
                        #calculate the density matrix from probabilities of each bit-string
                        rho_dict[bn][angle] = GraphState.calc_rho(pvecs)
            # if ignore neighbours
            else:
                rho_dict = {}
                for angle_i, pvecs in qst_pvecs[i].items():
                    angle = angles[int(angle_i[-1])]
                    rho_dict[angle] = GraphState.calc_rho(pvecs)#Note since calc_rho is not a self object under GraphState class so when
                                                                   #calling this function need to state GraphState class beforehand
            rho_dict_list.append(rho_dict)

        #if self.reps == 1:
        #    return rho_dict # rho_dict = {'(0,1)':{'00':4*4matrix,'01':4*4matrix,'10':4*4matrix,'11':4*4matrix},'(1,2)':{'00':..}}
                            #edge-adjacent qubits-density matrix structure

        return rho_dict_list
    
    def gen_entanglement_witness_intactness_circuits(self):
        """
        Generate entanglement witness (deprecated)
        """
        graphstate = self.delay_circuit.copy()
        circs = {}
        
        
        circMx = graphstate.copy('Mx')
        circMx.h(range(self.nqubits))
        classical_register_X = ClassicalRegister(self.nqubits)
        circMx.add_register(classical_register_X)
        circMx.measure(range(self.nqubits),classical_register_X)
        
        circMz = graphstate.copy('Mz')
        classical_register_Z = ClassicalRegister(self.nqubits)
        circMz.add_register(classical_register_Z)
        circMz.measure(range(self.nqubits),classical_register_Z)
        
        circs['Mx'] = circMx
        circs['Mz'] = circMz
        
        self.entanglement_witness_intactness_circuits = circs
        return circs
    
    def run_entanglement_witness_intactness_circuits(self, reps=1, shots=4096, qrem=False, sim=None):
        """
        run entanglement witness QST circuits (deprecated)
        """
        self.reps = reps
        self.shots = shots
        self.qrem = qrem
        self.sim = sim
        
        circ_list = []
        for circ in self.entanglement_witness_intactness_circuits.values():
            circ_list.append(circ)
        
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        
        circ_list = circ_list_multi
        
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)

        if sim is None:
            job = execute(circ_list, backend=self.backend, shots=shots)
        elif sim == "ideal":
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.nqubits)),
                          shots=shots)
        elif sim == "device":
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        return job
    
    def pvecs_Mx_Mz_from_result(self, result):
        """
        Get entanglement witness counts from qiskit result as dictionary or lists of dictionaries
        (deprecated)
        """
                
        if self.reps is None:
            try:  # Try to obtain QREM results
                result.get_counts('qrem0')
                self.qrem = True
            except:
                self.qrem = False
                
            if self.qrem is True:
                self.reps = int((len(result.results)-2)/2)#total number of witness circuits need to run/number of circuits(Mx&Mz) per rep
            else:                                                      
                self.reps = int(len(result.results)/2)
            self.shots = result.results[0].shots
            

        # Load counts as dict experiment-wise
        entanglement_witness_pvecs_multi = [] #[{'Mx':{'00':123,'01':211,...},'Mz':{'00':132,'01':282,...}}, {....}]
        for i in range(self.reps):
            n = self.nqubits
            entanglement_witness_pvecs = {'Mx': np.zeros(2**n), 'Mz': np.zeros(2**n)}
            
            Mx_counts = result.get_counts('Mx' + f'-{i}')
            for bit_str, count in Mx_counts.items():
                bit_str_binary = bit_str[::-1]
                idx = int(bit_str_binary, 2)
                entanglement_witness_pvecs['Mx'][idx] += count
            
            Mz_counts = result.get_counts('Mz' + f'-{i}')
            for bit_str, count in Mz_counts.items():
                bit_str_binary = bit_str[::-1]
                idx = int(bit_str_binary, 2)
                entanglement_witness_pvecs['Mz'][idx] += count
            entanglement_witness_pvecs_multi.append(entanglement_witness_pvecs)
            
        self.entanglement_witness_pvecs_multi = entanglement_witness_pvecs_multi
            
        # Save list of calibration matrices for each qubit
        if self.qrem is True:
            qrem_counts = [result.get_counts('qrem0'),
                           result.get_counts('qrem1')]

            M_list = [np.zeros((2, 2)) for i in range(self.nqubits)]
            for jj, counts in enumerate(qrem_counts):
                for bit_str, count in counts.items():
                    for i, q in enumerate(bit_str[::-1]):
                        ii = int(q)
                        M_list[i][ii, jj] += count

            # Normalise
            norm = 1/self.shots
            for M in M_list:
                M *= norm

            self.M_list = M_list

        return entanglement_witness_pvecs_multi  # Multiple experiments
    
    def apply_qrem_entanglement_witness_intactness(self, pvecs):
        """Apply quantum readout error mitigation on entanglement witness result counts/pvecs
        Can also be used to apply qrem to fidelity measurement of multi-qubit states
        
        Args:
            pvecs (list): list of probability vector dictionaries

        Returns:
            list: list of QREM prob vector dictionaries
        """
        pvecs_list = []

        for i in range(self.reps):
            pvecs_mit = pvecs[i].copy()

            for measure_basis, pvec in pvecs[i].items():
                # Invert n-qubit calibration matrix
                M_inv = la.inv(self.calc_M_multi(self.qubits_to_connect))
                # "Ideal" probability vector
                pvec_mit = np.matmul(M_inv, pvec)
                pvecs_mit[measure_basis] = pvec_mit

            pvecs_list.append(pvecs_mit)
            
        return pvecs_list
    
    def Mx_Mz_from_result(self, entanglement_witness_counts_pvecs):
        """
        obtain entanglement witness observables result (deprecated)
        """
        entanglement_witness_expected_values_list = []
        for i in range(self.reps):
            entanglement_witness_expected_values = {'Mx': 0, 'Mz': 0}
            Mx_pvec = entanglement_witness_counts_pvecs[i]['Mx']
            for idx in range(len(Mx_pvec)):
                n_ones = bin(idx).count('1')
                if n_ones % 2 == 0:
                    entanglement_witness_expected_values['Mx'] += Mx_pvec[idx]
                else:
                    entanglement_witness_expected_values['Mx'] -= Mx_pvec[idx]
            entanglement_witness_expected_values['Mx'] /= np.sum(Mx_pvec)
            
            Mz_pvec = entanglement_witness_counts_pvecs[i]['Mz']
            entanglement_witness_expected_values['Mz'] += Mz_pvec[0]
            entanglement_witness_expected_values['Mz'] += Mz_pvec[-1]
            entanglement_witness_expected_values['Mz'] /= np.sum(Mz_pvec)
        
            entanglement_witness_expected_values_list.append(entanglement_witness_expected_values)
            
        return entanglement_witness_expected_values_list   
    
    def find_entanglement_witness_intactness(self, result, apply_mit = False):
        """
        combine observables to entanglement witness (deprecated)
        """
        pvecs = self.pvecs_Mx_Mz_from_result(result)
        
        if apply_mit is True:
            pvecs = self.apply_qrem_entanglement_witness_intactness(pvecs)
        
        expected_values_list = self.Mx_Mz_from_result(pvecs)
        
        lower_bounds_list = []
        for i in range(self.reps):
            Mx = expected_values_list[i]['Mx']
            Mz = expected_values_list[i]['Mz']
            
            m_lower_bound = self.nqubits
            for m in range(2, self.nqubits+1)[::-1]:
                alpha = 2**(m-1)/(2**(m-1)-1)
                if alpha*Mz + Mx > alpha:
                    m_lower_bound = m-1
            lower_bounds_list.append(m_lower_bound)
        
        self.lower_bounds_list = lower_bounds_list
        
        return lower_bounds_list
    
    
    def gen_full_qst_circuits(self):
        """generate QST circuits for multi-qubit (>2) graph state

        Returns:
            dictionary: dictionary of qiskit circuits
        """
        #pauli-basis
        single_qst_basis = ['X','Y','Z']
        multiple_qubits_combinations = []
        #combine pauli-basis to pauli-strings
        for i in range(self.nqubits):
            multiple_qubits_combinations.append(single_qst_basis)
        multiple_qst_basis = [combination for combination in itertools.product(*multiple_qubits_combinations)]
        
        circuits = {}
        name_list = []
        #graphstate = self.delay_circuit.copy() #commented out when using delay circuits
        graphstate = self.circuit.copy()
        #graphstate.barrier()
        
        # change basis of each qubit to corresponding pauli-observable
        for basis in multiple_qst_basis:
            circ_name = ''.join(basis)
            circ = graphstate.copy(circ_name)
            name_list.append(circ_name)
            for i in range(len(basis)):
                if basis[i] == 'X':
                    circ.h(self.qubits_to_connect[i])
                elif basis[i] == 'Y':
                    circ.sdg(self.qubits_to_connect[i])
                    circ.h(self.qubits_to_connect[i])
            circuits[circ_name] = circ
        
        # add measurements to each circuits
        for circ in circuits.values():
            cr = ClassicalRegister(self.nqubits)
            circ.add_register(cr)
            circ.measure(self.qubits_to_connect, cr)
            #circ.measure_all()
        
        self.full_qst_circuits = circuits
        self.name_list = name_list
        
        return circuits
    
    def run_full_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None):
        """run full QST circuits (>2 qubits graph states)

        Args:
            reps (int, optional): number of repetitions. Defaults to 1.
            shots (int, optional): number of shots in each circuit. Defaults to 4096.
            qrem (bool, optional): whether execute with QREM circuits. Defaults to False.
            sim (str, optional): simulator mode. Defaults to None.

        Returns:
            IBMJob: job submitted to IBM Backend or onto simulators
        """
        self.reps = reps
        self.shots = shots
        self.qrem = qrem
        self.sim = sim

        # Convert circuits dict into list form
        circ_list = []
        for circuit in self.full_qst_circuits.values():
            circ_list.append(circuit)

        # Extend circuit list by number of repetitions
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        circ_list = circ_list_multi

        # Generate QREM circuits and append to circ_list if qrem == True
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)

        # If circuits are executed on a simulator or real backend
        if sim is None:
            job = execute(circ_list, backend=self.backend, shots=shots)
        elif sim == "ideal":
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.device_size)),
                          shots=shots)
        elif sim == "device":
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        return job  
    
    def full_qst_pvecs_from_result(self, result):
        """results from full QST circuits

        Args:
            result (IBMJob.result()): job result

        Returns:
            lists of dictionaries: counts and probability vectors
        """
        #when run this make sure self.gen_full_qst_circuits is executed beforehand
        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))#total number of circuits need to run/ number of qst circuits per rep
                                                                    #s.t. all pairs are fulled QST
            self.shots = result.results[0].shots
            try:  # Try to obtain QREM results
                result.get_counts('qrem0')
                self.qrem = True
            except:
                self.qrem = False
                
        # Load counts as dict experiment-wise
        full_qst_pvecs_multi = [] #[{'XX':[0.1 0.2...],'XY':[0.2 0.2...]}, {'XX':[0.1 0.1...],'XY':[0.2 0.2...]}, ...]
        full_qst_counts_multi = []
        for i in range(self.reps):
            n = self.nqubits
            full_qst_pvecs = {basis: np.zeros(2**n) for basis in self.name_list}
            full_qst_counts ={basis: 0 for basis in self.name_list}
            for name in self.name_list:
                name_ext = name + f'-{i}'
                counts = result.get_counts(name_ext)#get counts of each circuit from result
                full_qst_counts[name] = counts
                for bit_str, count in counts.items():
                    bit_str_binary = bit_str[::-1] #conert back to top-ending notation
                    idx = int(bit_str_binary, 2) #convert bit-string to index in pvec
                    full_qst_pvecs[name][idx] += count
                full_qst_pvecs[name] /= self.shots
                    
            full_qst_pvecs_multi.append(full_qst_pvecs)
            full_qst_counts_multi.append(full_qst_counts)
        self.full_qst_pvecs_multi = full_qst_pvecs_multi
        self.full_qst_counts_multi = full_qst_counts_multi
        
        # Save list of calibration matrices for each qubit
        if self.qrem is True:
            qrem_counts = [result.get_counts('qrem0'),
                           result.get_counts('qrem1')]

            M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
            for jj, counts in enumerate(qrem_counts):
                for bit_str, count in counts.items():
                    for i, q in enumerate(bit_str[::-1]):
                        ii = int(q)
                        M_list[i][ii, jj] += count

            # Normalise
            norm = 1/self.shots
            for M in M_list:
                M *= norm

            self.M_list = M_list

        return full_qst_pvecs_multi, full_qst_counts_multi  # Multiple experiments
    
    def apply_mthree(self, pvecs_list, counts_list, mit=None):
        """apply M3 mitigation to counts/pvecs

        Args:
            pvecs_list (list): list of probability vectors dictionaries
            counts_list (list): list of counts dictionaries
            
        Returns:
            two lists of dictionaries: mitigated pvecs and counts
        """
        #load the calibration data
        if mit is None:
            mit = load_cal(self.backend)
        self.mit = mit
        
        counts_list_mit = counts_list.copy()
        pvecs_list_mit = pvecs_list.copy()
        
        for i in range(self.reps):
            for basis, counts in counts_list[i].items():
                #apply correction to the counts obtained with specified qubits
                corrected_counts = self.mit.apply_correction(counts, self.qubits_to_connect)
                counts_list_mit[i][basis] = corrected_counts.nearest_probability_distribution()
                #Unnormalise/store data into corrected pvecs
                for bit_str, prob in counts_list_mit[i][basis].items():
                    counts_list_mit[i][basis][bit_str] = prob*self.shots
                    bit_str_binary = bit_str[::-1]
                    idx = int(bit_str_binary, 2)
                    pvecs_list_mit[i][basis][idx] = prob
        
        return pvecs_list_mit, counts_list_mit
    
    def recon_full_density_mats(self, result, apply_mit = 'M3'):
        """reconstruct 2^N * 2^N density matrices of N-qubit graph state

        Args:
            result (IBMJob.result()): Job result
            apply_mit (str, optional): mitigation mode. Defaults to 'M3'.

        Returns:
            list: list of density matices in different repetitions 
        """
        rho_list = []
        #obtain probabilities vectors from each QST basis and each repitition
        pvecs, counts = self.full_qst_pvecs_from_result(result)

        # If mitigation is applied
        if apply_mit == 'qrem':
            pvecs = self.apply_qrem_entanglement_witness_intactness(pvecs)
        elif apply_mit == 'M3':
            pvecs, _ = self.apply_mthree(pvecs, counts)
        
        single_qst_basis_ext = ['I','X','Y','Z']
        multiple_qubits_combinations = []
        
        #establish expected values dict: s_dict = {'II':1, 'IX':0, 'IY':0,...., 'ZZ':...}
        for i in range(self.nqubits):
            multiple_qubits_combinations.append(single_qst_basis_ext)
        s_dict = {''.join(combination): 0. for combination in itertools.product(*multiple_qubits_combinations)}
        
        mapping = {}
        #iterate over the pauli-string basis
        for basis in s_dict.keys():
            if basis.count('I') > 0:
                for pauli_string in self.name_list:
                    #create a mapping of identity-contained pauli string can be measured by no-identity pauli strings
                    if self.is_measured_by(basis, pauli_string):
                        mapping[basis] = pauli_string
                        break
        # Reconstruct density matrices for each repitition
        for i in range(self.reps):
            n = self.nqubits
            rho = np.zeros([2**n, 2**n], dtype=complex)
            #load information of expected values from non-identity basis
            for basis, pvec in pvecs[i].items():
                for idx in range(len(pvec)):
                    n_ones = bin(idx).count('1') #count number of 1s appeared in a specific bit string measured
                    # if odd 1s, negative contribution (product = -1), if even 1s, positive contribution (product = +1)
                    if n_ones % 2 == 0:
                        s_dict[basis] += pvec[idx]
                    else:
                        s_dict[basis] -= pvec[idx]
                   
            #load information of expected values from identity-included basis
            for basis in s_dict.keys():
                if basis.count('I') > 0:
                    measure_basis = mapping[basis]
                    pvec = pvecs[i][measure_basis]
                    s_dict[basis] = self.calc_expected_value(pvec, basis)
                    
                    #identities_idx = [i for i in range(len(basis)) if basis[i] == 'I']#positions in basis where Identity appear
                    #non_identities_paulis_list = [basis[i] for i in range(len(basis)) if basis[i] != 'I']
                    #non_identities_paulis = ''.join(non_identities_paulis_list)#basis in string with identities removed
                    
                    # find sub basis strings after cutting out the positions where identities appeared
                    #sub_name_list = self.name_list.copy()
                    #for s in range(len(sub_name_list)):
                    #    sub_name_list[s] = list(map(str, sub_name_list[s]))
                    #    new_sub_name_list = []
                    #    for j in range(len(sub_name_list[s])):
                    #        if j not in identities_idx:
                    #            new_sub_name_list.append(sub_name_list[s][j])
                    #    sub_name_list[s] = new_sub_name_list
                    #    sub_name_list[s] = ''.join(sub_name_list[s])
                    
                    # find all non-identity basis where its sub basis string matches the sub basis with identity
                    #sub_basis = [self.name_list[i] for i in range(len(self.name_list)) if sub_name_list[i] == non_identities_paulis]
                    # take the average to simulate the expected value for this basis
                    #s_dict[basis] = sum([s_dict[sub] for sub in sub_basis])/len(sub_basis)
            s_dict['I'*self.nqubits] = 1.  # S for 'II' always equals 1
            
            for basis, s in s_dict.items():
                rho += (1/(2**self.nqubits))*s*pauli_n(basis) 
                
            rho = GraphState.find_closest_physical(rho)    
            rho_list.append(rho)
            
        return rho_list
    
    def is_measured_by(self, pauli_a, pauli_b):
        """check if pauli_a (bit-string) can be measured by pauli_b (bit-string)
        True if pauli_a is bit-wise commuting with pauli_b and has more or same number of identities

        Args:
            pauli_a (str): bit-string
            pauli_b (str): bit-string measured by

        Returns:
            bool: true if a can be measured by b, vice versa
        """
        for i in range(len(pauli_a)):
            if pauli_a[i] != pauli_b[i]:
                if pauli_a[i] != 'I':
                    return False
        return True
    
    def calc_expected_value(self, pvec, pauli_string):
        """calculate expected values of the probability vector w.r.t the QST basis

        Args:
            pvec (numpy array): 2^N dimension array of probability vector
            pauli_string (str): pauli-string of this QST basis

        Returns:
            float: expected value of this QST basis
        """
        expected_val = 0
        identity_idx = []
        #Find the indicies where Identity appears in pauli string
        for i in range(len(pauli_string)):
            if pauli_string[i] == 'I':
                identity_idx.append(i)
            
        #Calculate the expected value of the pauli string measurement <pauli_string>
        for idx in range(len(pvec)):
            bit_str = bin(idx)[2:].zfill(self.nqubits)
            one_count = 0 #count number of 1s appeared in a specific bit string measured
                          # if odd 1s, negative contribution (product = -1), if even 1s, positive contribution (product = +1)
            for i in range(len(bit_str)):
                #count number of 1s while ignore if it was measured from Identity
                if bit_str[i] == '1' and not (i in identity_idx):
                    one_count += 1
            if one_count % 2 == 0:
                expected_val += pvec[idx]
            else:
                expected_val -= pvec[idx]
        return expected_val
    
    def find_fidelities_to_GraphState(self, rho_list):
        """Calculate the fidelities of the fully qst Graph state to the ideal Graph state

        Args:
            rho_list (list): list of density matrices

        Returns:
            list: list of fidelities
        """
        #create a mapping of the qubit index in IBM machine to its order of measurement
        index_dict = {}
        for i in range(len(self.qubits_to_connect)):
            index_dict[self.qubits_to_connect[i]] = self.nqubits-1-i
        
        graphstate = QuantumCircuit(self.nqubits)
        unconnected_edges = self.edge_list.copy()
        # Apply Hadamard gates to every qubit
        graphstate.h(list(range(self.nqubits)))
        # Connect every edge with cz gates in same topology as the real machine 
        # (and in same order when circuit is created on that machine)
        while unconnected_edges:
            connected_qubits = []  # Qubits already connected in the current time step
            connected_edges = []
            remove = []
            for edge in unconnected_edges:
                if np.any(np.isin(edge, connected_qubits)) == False:
                    #make sure cz connects the qubits in same topology as (edge[0],edge[1]) does in real machine
                    graphstate.cz(index_dict[edge[0]], index_dict[edge[1]])
                    connected_qubits.extend(edge)
                    remove.append(edge)
            # Remove connected edges from unconnected edges list
            for edge in remove:
                unconnected_edges.remove(edge)
       
        #graphstate = QuantumCircuit(self.nqubits)
        #graphstate.h(list(range(self.nqubits)))
        #unconnected_edges = self.edge_list.copy()
        #for edge in unconnected_edges:
        #    control = index_dict[edge[0]]
        #    target = index_dict[edge[1]]
        #    graphstate.cz(control, target)

        #from previous circuit, find the corresponding ideal density matrices
        graphstate_ideal = Statevector(graphstate)
        rho_ideal = DensityMatrix(graphstate_ideal).data

        overlaps = np.matmul(rho_list, rho_ideal)
        #by definition F = ||<a|b>||^2 = Tr(|a><a|b><b|) and rho = |a><a|, rho_ideal = |b><b|
        fidelities = []
        for overlap in overlaps:
            fidelity = np.trace(overlap)
            fidelities.append(fidelity)
        
        self.fidelities = fidelities
        return fidelities
        
        
        
    def run_qst(self, reps=1, shots=4096, qrem=False, sim=None, output='default',
                execute_only=False, execution_mode='execute'):
        """Run entire QST program to obtain qubit pair density matrices with
        option to only send job request

        Args:
            reps (int, optional): number of repetitions of circuit list. Defaults to 1.
            shots (int, optional): number of shots per circuit. Defaults to 4096.
            qrem (bool, optional): whether execute QREM circuit. Defaults to False.
            sim (_type_, optional): simulator mode. Defaults to None.
            output (str, optional): output mode. Defaults to 'default'.
            execute_only (bool, optional): whether execute only or construct the density matrices. Defaults to False.
            execution_mode (str, optional): executation mode (execute/transpile). Defaults to 'execute'.

        Returns:
            IBMJob or dictionary: job submitted or density matrices dictionary
        """
        self.gen_qst_circuits()
        job = self.run_qst_circuits(reps, shots, qrem, sim, execution_mode)

        if execute_only is True:  # If only executing job
            return job

        # Otherwise obtain jobs results
        result = job.result()
        rho_dict = self.qst_from_result(result, output)
        return rho_dict

    def qst_from_result(self, result, output='default'):
        """Process Qiskit Result into qubit pair density matrices. Can be used
        with externally obtained results.

        Args:
            result (IBMJob.result()): job submitted/completed
            output (str, optional): wheter output mitigated density matrices teogether. Defaults to 'default'.

        Returns:
            dictionaries: dictionaries of density matrices
        """
        # If QST circuits haven't been generated
        if self.qst_circuits is None:
            self.gen_qst_circuits()

        # Output only mitigated result if self.qrem is True or only unmitigated
        # result if self.qrem is False
        if output == 'default':
            rho_dict = self.recon_density_mats(result, apply_mit=None)
            return rho_dict
        # No mitigation
        if output == 'nomit':
            rho_dict = self.recon_density_mats(result, apply_mit=False)
            return rho_dict
        # Output both mitigated and unmitigated results
        if output == 'all':
            rho_dict = self.recon_density_mats(result, apply_mit=False)
            rho_dict_mit = self.recon_density_mats(result, apply_mit=True)
            return rho_dict_mit, rho_dict

        return None

    def gen_batches(self):
        """Get a dictionary of tomography batches, where keys are batch numbers
        and values are lists of tomography groups (targets + adj qubits).
        QST can be performed on batches in parallel.

        Returns:
            dictionary: dictionary of batches (key are batch name, values are group of qubits in the batch)
        """
        batches = {}
        group_list = []

        unbatched_edges = self.edge_list.copy()
        i = 0
        # Loop over unbatched edges until no unbatched edges remain
        while unbatched_edges:
            batches[f'batch{i}'] = []
            batched_qubits = []
            remove = []

            for edge in unbatched_edges:
                group = tuple(list(edge) + self.adj_qubits[edge])
                # Append edge to batch only if target and adjacent qubits have
                # not been batched in the current cycle
                if np.any(np.isin(group, batched_qubits)) == False:
                    batches[f'batch{i}'].append(group)
                    group_list.append(group)

                    batched_qubits.extend(group)
                    remove.append(edge)

            for edge in remove:
                unbatched_edges.remove(edge)
            i += 1

            self.batches = batches #self.batches = {'batch1':[(0,1,2),(4,5,3,6)],'batch2':[...]} each batch can run qst circuit in parallel 
            self.group_list = group_list

        return batches
    
    def bit_str_to_BellState(bit_str):
        """
        Deprecated function no longer used, see recent version in teleportation.py
        """
        #Base BellState
        state_if_before_processing = BS_list_even[0].copy()

        bit_str_new = bit_str[::-1]
        for i in range(len(bit_str_new)):
            state_if_before_processing.h(1)
            if bit_str_new[i] == '1':
                state_if_before_processing.x(1)
        
        
        state_column = Statevector(state_if_before_processing)
        BellState_index = 0
        for i in range(4):
            if len(bit_str) % 2 == 0:
                if state_column.equiv(States_even[i]):
                    BellState_index = i+1
            else:
                if state_column.equiv(States_odd[i]):
                    BellState_index = i+1
        return f'BS_{BellState_index}'
    
    def gen_qst_circuits(self):
        """Generates (parallelised) quantum state tomography circuits

        Returns:
            dictionary: dictionary of circuits, in the batch/basis/circuit structure 
        """
        # Generate batches of groups (target edges + adjacent qubits) to perform
        # QST in parallel
        if self.batches is None:
            self.batches = self.gen_batches()

        circuits = {}  # Dictionary of groups of circuits where batches are keys
        name_list = []  # List of circuit names

        graphstate = self.circuit.copy()
        #graphstate = self.delay_circuit.copy() #commented out when using delay circuits
        graphstate.barrier()

        for batch, groups in self.batches.items():

            # Dictionary of circuits where measurement basis are keys
            batch_circuits = {}

            # Nx2 array of target (first two) edges
            targets = [g[:2] for g in groups]
            targ_array = np.array(targets)
            flat_array = targ_array.flatten()

            # Create circuits for each basis combination over target pairs
            circxx = graphstate.copy(batch + '-' + 'XX')
            circxx.h(flat_array)
            batch_circuits['XX'] = circxx

            circxy = graphstate.copy(batch + '-' + 'XY')
            circxy.sdg(targ_array[:, 1].tolist())
            circxy.h(flat_array)
            batch_circuits['XY'] = circxy

            circxz = graphstate.copy(batch + '-' + 'XZ')
            circxz.h(targ_array[:, 0].tolist())
            batch_circuits['XZ'] = circxz

            circyx = graphstate.copy(batch + '-' + 'YX')
            circyx.sdg(targ_array[:, 0].tolist())
            circyx.h(flat_array)
            batch_circuits['YX'] = circyx

            circyy = graphstate.copy(batch + '-' + 'YY')
            circyy.sdg(flat_array)
            circyy.h(flat_array)
            batch_circuits['YY'] = circyy

            circyz = graphstate.copy(batch + '-' + 'YZ')
            circyz.sdg(targ_array[:, 0].tolist())
            circyz.h(targ_array[:, 0].tolist())
            batch_circuits['YZ'] = circyz

            circzx = graphstate.copy(batch + '-' + 'ZX')
            circzx.h(targ_array[:, 1].tolist())
            batch_circuits['ZX'] = circzx

            circzy = graphstate.copy(batch + '-' + 'ZY')
            circzy.sdg(targ_array[:, 1].tolist())
            circzy.h(targ_array[:, 1].tolist())
            batch_circuits['ZY'] = circzy

            circzz = graphstate.copy(batch + '-' + 'ZZ')
            batch_circuits['ZZ'] = circzz

            for circ in batch_circuits.values():
                name_list.append(circ.name)
                # Create a seperate classical register for each group in batch
                # and apply measurement gates respectively
                for group in groups:
                    cr = ClassicalRegister(len(group))
                    circ.add_register(cr)
                    circ.measure(group, cr)

            circuits[batch] = batch_circuits #circuits['batch1']['ZZ'] corresponds 
                                             #to a qiskit circuit object with Graph state+QST circuit ZZ

            self.qst_circuits = circuits
            self.name_list = name_list

        return circuits        
    
    def run_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None, execution_mode='execute'):
        """Execute the quantum state tomography circuits

        Args:
            reps (int, optional): number of QST circuits repetitions. Defaults to 1.
            shots (int, optional): number of shots per circuit. Defaults to 4096.
            qrem (bool, optional): whether add QREM circuit to run. Defaults to False.
            sim (_type_, optional): simulator mode (noisless/noisemodel/real device). Defaults to None.
            execution_mode (str, optional): execution mode (execute/transpile/sampler). Defaults to 'execute'.

        Returns:
            IBMJob: job submitted to IBM
        """
        self.reps = reps
        self.shots = shots
        self.qrem = qrem
        self.sim = sim

        # Convert circuits dict into list form
        circ_list = []
        for batch in self.qst_circuits.values():
            for circuit in batch.values():
                circ_list.append(circuit)

        # Extend circuit list by number of repetitions
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        circ_list = circ_list_multi

        # Generate QREM circuits and append to circ_list if qrem == True
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)

        # If circuits are executed on a simulator or real backend or runtime sampler mode
        if sim is None:
            if execution_mode == 'execute':
                job = execute(circ_list, backend=self.backend, shots=shots)
            elif execution_mode == 'transpile':
                circ_list_transpiled = transpile(circ_list, backend = self.backend)
                job = self.backend.run(circ_list_transpiled, shots=shots)
            elif execution_mode == 'runtime_sampler':
                service = QiskitRuntimeService()
                backend = service.backend(self.backend.name)
                options = Options()
                options.execution.shots = shots
                options.max_execution_time = 7200 #check this later
                options.optimization_level = 1

                with Session(service=service, backend=backend) as session:
                    sampler = Sampler(session=session, options=options)
                    job = sampler.run(circ_list)
                
        elif sim == "ideal":
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.nqubits)),
                          shots=shots)
        elif sim == "device":
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        return job
            
    def counts_from_result(self, result):
        """Get counts from qiskit result as dictionary or lists of dictionaries

        Args:
            result (IBMJob.result()): job result completed

        Returns:
            list: list of dictionary of counts
        """
        #obtain the reps, shots and qrem from the result as if we don't know them
        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))# number of qubits？/total number of QST circuits need to run 
                                                                    #s.t. all pairs are fulled QST
            #self.reps = int(len(result.quasi_dists)/len(self.name_list))
            self.shots = result.results[0].shots
            #self.shots = result.metadata[0]['shots']
            try:  # Try to obtain QREM results
                result.get_counts('qrem0')
                self.qrem = True
            except:
                self.qrem = False

        # Load counts as dict experiment-wise
        qst_counts_multi = []
        for i in range(self.reps):
            qst_counts = {batch: {} for batch in self.batches.keys()}
            for name in self.name_list:
                batch, basis = name.split('-')
                name_ext = name + f'-{i}'
                qst_counts[batch][basis] = result.get_counts(name_ext) #e.g. qst_counts['batch1']['ZZ'] = {'0000':21,'0001':12,....}
            qst_counts_multi.append(qst_counts) #e.g. qst_counts_multi = [{qst_counts 1st iter}, {qst_counts 2nd iter},...]

        # Save list of calibration matrices for each qubit
        if self.qrem is True:
            qrem_counts = [result.get_counts('qrem0'),
                           result.get_counts('qrem1')]

            M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
            for jj, counts in enumerate(qrem_counts):
                for bit_str, count in counts.items():
                    for i, q in enumerate(bit_str[::-1]):
                        ii = int(q)
                        M_list[i][ii, jj] += count

            # Normalise
            norm = 1/self.shots
            for M in M_list:
                M *= norm

            self.M_list = M_list
        
        #qst_counts_multi = []
        #x = 0
        #for i in range(self.reps):
        #    qst_counts = {batch: {} for batch in self.batches.keys()}
        #    for name in self.name_list:
        #        batch, basis = name.split('-')
        #        distribution = result.quasi_dists[x] #e.g. qst_counts['batch1']['ZZ'] = {'0000':21,'0001':12,....}
        #        counts_bit_str = {}
        #        for integer, prob in distribution.items():
        #            counts_bit_str[bin(integer)[2:]] = prob*self.shots
        #        qst_counts[batch][basis] = counts_bit_str
        #        x += 1
        #    qst_counts_multi.append(qst_counts) #e.g. qst_counts_multi = [{qst_counts 1st iter}, {qst_counts 2nd iter},...]

        # Save list of calibration matrices for each qubit
        #if self.qrem is True:
        #    qrem_counts = [result.quasi_dists[-2], result.quasi_dists[-1]]

        #    M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
        #    for jj, distribution in enumerate(qrem_counts):
        #        for integer, prob in distribution.items():
        #            bit_str = bin(integer)[2:]
        #            for i, q in enumerate(bit_str[::-1]):
        #                ii = int(q)
        #                M_list[i][ii, jj] += prob

        #    self.M_list = M_list
        
        if self.reps == 1:
            return qst_counts  # Single experiment

        return qst_counts_multi  # Multiple experiments

    def group_counts(self, qst_counts):
        """Regroups qst_counts according to tomography groups (target qubit pair
        + adjacent qubits) and convert into equivalent probability vector

        Args:
            qst_counts (list): list of counts dictionaries

        Returns:
            two lists: lists of dictionaries of counts and pvecs, in rep-group-basis-count structure
        """
        g_counts_list = []  # Regrouped counts
        g_vecs_list = []  # Equivalent probability vectors

        if isinstance(qst_counts, dict):
            #check if qst_counts is a dictationary object, if it is put it into an list just like qst_counts_multi
            qst_counts = [qst_counts]

        for i in range(self.reps):
            # Construct dictionary keys
            g_counts = {}
            g_vecs = {}
            for group in self.group_list:
                n = len(group)
                g_counts[group] = {basis: {bit_str: 0.
                                           for bit_str in bit_str_list(n)}
                                   for basis in basis_list}
                #expect to have g_counts = {'(0,1,2)':{'XX':{'000':0.,'001':0.},'XY':{'000':0.,'001':0.}}, '(4,5,3,6)':{....}}
                g_vecs[group] = {basis: np.zeros(2**n) for basis in basis_list}
                #expect to have g_vecs = {'(0,1,2)':{'XX':[0 0 0...],'XY':[0 0 0...]..}, '(4,5,3,6)':{....}}

            # Nested loop over each bit string over each basis over each batch in
            # raw counts obtained from self.counts_from_result()
            for batch, batch_counts in qst_counts[i].items():
                for basis, counts in batch_counts.items():
                    for bit_str, count in counts.items():
                        # Reverse and split bit string key in counts
                        split = bit_str[::-1].split()
                        # Loop over every group in batch and increment corresponding
                        # bit string counts
                        for ii, group in enumerate(self.batches[batch]):
                            #originally have [(0,1,2),(3,4,5,6)], enumerate returns [(0,(0,1,2)),(1,(3,4,5,6))]
                            # so ii, group will be 0, (0,1,2) & 1, (3,4,5,6) respectively
                            g_counts[group][basis][split[ii]] += count
                            g_vecs[group][basis][int(split[ii], 2)] += count

            g_counts_list.append(g_counts) #number of counts under each Tomography basis with corresponding group and iteration number
            # the main job of this part is to convert qst_counts from batch-basis-counts structure to batch-group-basis-counts structure
            # so that we know which pair of qubits the counts are referring to
            g_vecs_list.append(g_vecs)

        if self.reps == 1:
            return g_counts, g_vecs  # Single experiment

        return g_counts_list, g_vecs_list  # Multiple experiments

    def bin_pvecs(self, g_counts):
        """Further classify the group probability vectors according to the different
        measurement combinations on adjacent qubits

        Args:
            g_counts (list): list of counts dictionaries

        Returns:
            list: list of categorised counts dictionaries, in the rep-edge-bin-basis-pvec structure
        """
        b_pvecs_list = []

        if isinstance(g_counts, dict):
            g_counts = [g_counts]

        for i in range(self.reps):
            b_pvecs = {}
            for edge in self.edge_list:
                n = len(self.adj_qubits[edge])
                b_pvecs[edge] = {bn: {basis: np.zeros(4)
                                      for basis in basis_list}
                                 for bn in bit_str_list(n)}
                #b_pvecs={(0,1):{'00':{'XX':[0 0 0 0],'XY':[0 0 0 0],...},'01':{...},...}},(1,2):}
                #where '00', '01' are the adjacent qubits results of each edge, note the pvec [0 0 0 0] is not for adjacent qubits but for
                #the edge itself (that's why it's np.zeros(4) not other numbers)

            for group, basis_counts in g_counts[i].items():
                edge = group[:2]
                for basis, counts in basis_counts.items():
                    for bit_str, count in counts.items():
                        idx = int(bit_str[:2], 2) #only take the first 2 index in bit_str since edge is always in the first 2
                        bn = bit_str[2:]#the remaining bit_str must be the adjacent qubits to this edge

                        b_pvecs[edge][bn][basis][idx] += count
                        #probability[edge][measured adjacent qubits][tomography basis][counts of this edge]

                    # Normalise
                    for bn in bit_str_list(len(self.adj_qubits[edge])):
                        pvec = b_pvecs[edge][bn][basis]
                        norm = 1/pvec.sum()
                        b_pvecs[edge][bn][basis] = pvec*norm

            b_pvecs_list.append(b_pvecs)

        if self.reps == 1:
            return b_pvecs

        return b_pvecs_list

    def recon_density_mats(self, result, apply_mit=None):
        """Reconstruct the density matrices for every qubit pair for every
        measurement combination and save it as a dictionary or list of
        dictionaries

        Args:
            result (IBMJob.result()): job result completed
            apply_mit (bool, optional): whether apply QREM. Defaults to None.

        Returns:
            list: list of density matrices
        """
        rho_dict_list = []

        qst_counts = self.counts_from_result(result)# obtain raw data in the batch form
        g_counts, g_vecs = self.group_counts(qst_counts)# obtain processed data in group-basis-counts form

        # Whether mitigation is applied or not defaults to whether qrem circuits
        # are included in result
        if apply_mit is None:
            apply_mit = self.qrem
        # If mitigation is applied
        if apply_mit is True:
            g_counts, g_vecs = self.apply_qrem(g_counts, g_vecs)#convert raw counts into error mitigated counts
                                                                #Note M matrices are already calculated in counts_from result function
                                                                #which are then used in apply_qrem

        b_pvecs = self.bin_pvecs(g_counts)#convert data into edge-adjacent-qubits-basis-counts form vector

        if isinstance(b_pvecs, dict):
            b_pvecs = [b_pvecs]

        for i in range(self.reps):
            rho_dict = {edge: {} for edge in self.edge_list}
            for edge, bns in b_pvecs[i].items():
                for bn, pvecs in bns.items():
                    rho_dict[edge][bn] = GraphState.calc_rho(pvecs)#Note since calc_rho is not a self object under GraphState class so when
                                                                   #calling this function need to state GraphState class beforehand
            rho_dict_list.append(rho_dict)

        if self.reps == 1:
            return rho_dict # rho_dict = {'(0,1)':{'00':4*4matrix,'01':4*4matrix,'10':4*4matrix,'11':4*4matrix},'(1,2)':{'00':..}}
                            #edge-adjacent qubits-density matrix structure

        return rho_dict_list

    def gen_qrem_circuits(self):
        """Generate QREM circuits

        Returns:
            list: list of two circuits from QREM
        """
        #circ0 = QuantumCircuit(self.device_size, name='qrem0')
        #circ0.measure_all()

        #circ1 = QuantumCircuit(self.device_size, name='qrem1')
        #circ1.x(range(self.device_size))
        #circ1.measure_all()
        
        properties = self.backend.properties()
        faulty_qubits = properties.faulty_qubits()
        qubits_to_measure = list(range(self.device_size))
        for q in faulty_qubits:
            qubits_to_measure.remove(q)
            
        circ0 = QuantumCircuit(self.device_size, self.device_size, name='qrem0')
        circ0.measure(qubits_to_measure, qubits_to_measure)

        circ1 = QuantumCircuit(self.device_size, self.device_size, name='qrem1')
        circ1.x(qubits_to_measure)
        circ1.measure(qubits_to_measure, qubits_to_measure)
        
        self.qrem_circuits = [circ0, circ1]

        return [circ0, circ1]

    def apply_qrem(self, g_counts, g_vecs):
        """Apply quantum readout error mitigation on grouped counts/probability
        vectors

        Args:
            g_counts (list): list of counts dictionaries
            g_vecs (list): list of pvec dictionaries

        Returns:
            two lists: same as input, but error mitigated
        """
        g_counts_list = []
        g_vecs_list = []

        if isinstance(g_counts, dict):
            g_counts = [g_counts]
            g_vecs = [g_vecs]

        for i in range(self.reps):
            g_counts_mit = g_counts[i].copy()
            g_vecs_mit = g_vecs[i].copy()

            for group, vecs in g_vecs[i].items():
                n = len(group)
                # Invert n-qubit calibration matrix
                M_inv = la.inv(self.calc_M_multi(group))
                for basis, vec in vecs.items():
                    # "Ideal" probability vector
                    vec_mit = np.matmul(M_inv, vec)
                    g_vecs_mit[group][basis] = vec_mit
                    # Equivalent ideal group counts
                    for ii, count in enumerate(vec_mit):
                        bit_str = bin(ii)[2:].zfill(n)
                        g_counts_mit[group][basis][bit_str] = count

            g_counts_list.append(g_counts_mit)
            g_vecs_list.append(g_vecs_mit)

        if self.reps == 1:
            return g_counts_mit, g_vecs_mit
            #g_counts_mit and g_vecs_mit are exactly same structure as g_counts and g_vecs in qst_group function BUT error mitigated

        return g_counts_list, g_vecs_list

    def calc_M_multi(self, qubits):
        """Compose n-qubit calibration matrix by tensoring single-qubit matrices

        Args:
            qubits (list): list of qubits indecies

        Returns:
            numpy2d array: calibration matrix on those qubits
        """
        M = self.M_list[qubits[0]]
        for q in qubits[1:]:
            M_new = np.kron(M, self.M_list[q])
            M = M_new

        return M

    @staticmethod
    def calc_rho(pvecs):
        """Calculate density matrix from probability vectors

        Args:
            pvecs (dictionary): key are the basis, values are pvec

        Returns:
            numpy2d array: list of density matrices
        """
        rho = np.zeros([4, 4], dtype=complex)

        # First calculate the Stokes parameters s
        s_dict = {basis: 0. for basis in ext_basis_list}
        s_dict['II'] = 1.  # S for 'II' always equals 1

        # Calculate s in each experimental basis
        for basis, pvec in pvecs.items():
            # s for basis not containing I
            s_dict[basis] = pvec[0] - pvec[1] - pvec[2] + pvec[3]
            # s for basis 'IX' and 'XI' (and other paulis)
            s_dict['I' + basis[1]] += (pvec[0] - pvec[1] + pvec[2] - pvec[3])/3 #+ or - is only decided by whether 2nd qubit is measured
                                                                                #to be |0> or |1> because only identity is applied to 1st
                                                                                #qubit so its result is not important
            s_dict[basis[0] + 'I'] += (pvec[0] + pvec[1] - pvec[2] - pvec[3])/3
            
        # Weighted sum of basis matrices
        for basis, s in s_dict.items():
            rho += 0.25*s*pauli_n(basis)

        # Convert raw density matrix into closest physical density matrix using
        # Smolin's algorithm (2011)
        rho = GraphState.find_closest_physical(rho)

        return rho

    @staticmethod
    def find_closest_physical(rho):
        """Algorithm to find closest physical density matrix from Smolin et al.

        Args:
            rho (numpy2d array): (unphysical) density matrix

        Returns:
            numpy2d array: physical density matrix
        """
        rho = rho/rho.trace()
        rho_physical = np.zeros(rho.shape, dtype=complex)
        # Step 1: Calculate eigenvalues and eigenvectors
        eigval, eigvec = la.eig(rho)
        # Rearranging eigenvalues from largest to smallest
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval_new = np.zeros(len(eigval), dtype=complex)

        # Step 2: Let i = number of eigenvalues and set accumulator a = 0
        i = len(eigval)
        a = 0

        while (eigval[i-1] + a/i) < 0:
            a += eigval[i-1]
            i -= 1

        # Step 4: Increment eigenvalue[j] by a/i for all j <= i
        # Note since eigval_new is initialized to be all 0 so for those j>i they are already set to 0
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            # Step 5 Construct new density matrix
            rho_physical += eigval_new[j] * \
                np.outer(eigvec[:, j], eigvec[:, j].conjugate()) #rho = Sum(lambdai*|lambdai><lambdai|)

        return rho_physical

def calc_negativities(rho_dict, mode='all'):
    """Obtain negativities corresponding to every density matrix in rho_dict.
    Option to obtain max, mean or all negativities between measurement
    combinations (bins)

    Args:
        rho_dict (dictionary): dictionary of density matrix of each edge
        mode (str, optional): _description_. Defaults to 'all'.

    Returns:
        list or dictionary: negativities calculated from density matrices
    """

    n_all_list = []  # Negativities for each bin per experiment
    n_mean_list = []  # Mean negativity between bins per experiment
    n_max_list = []  # Max negativity between bins per experiment
    n_min_list = []

    if isinstance(rho_dict, dict):
        rho_dict = [rho_dict]

    nexp = len(rho_dict)

    for i in range(nexp):
        n_all = {edge: {} for edge in rho_dict[i].keys()}# n_all = {'(0,1)'{'00':0.3,'01':0.4,'10':0.4...},'(1,2)':{'00':0.4,'01':...}}
        n_mean = {}
        n_max = {}
        n_min = {}

        for edge, bns in rho_dict[i].items():
            n_sum = 0.
            n_list = []

            for bn, rho in bns.items():
                n = calc_n(rho)

                n_all[edge][bn] = n
                n_sum += n #total negativities of this specific pair of qubits
                n_list.append(n)

            n_mean[edge] = n_sum/len(bns)# n_mean = {'(0,1)':0.42,'(1,2)':0.23,...} which is the average negativities of each pair of qubits
            n_max[edge] = max(n_all[edge].values())#n_max is the maximum negativity of each pair of qubit
            n_min[edge] = min(n_all[edge].values())#n_min is the minimum negativity of each pair of qubit

        n_all_list.append(n_all)
        n_mean_list.append(n_mean)
        n_max_list.append(n_max)
        n_min_list.append(n_min)

    # Single experiment
    if len(rho_dict) == 1:
        if mode == 'all':
            return n_all
        elif mode == 'mean':
            return n_mean
        elif mode == 'max':
            return n_max
        elif mode == 'min':
            return n_min

    # Multiple experiments
    if mode == 'all':
        return n_all_list
    elif mode == 'mean':
        return n_mean_list
    elif mode == 'max':
        return n_max_list
    elif mode == 'min':
        return n_min_list

    return None

def calc_entanglement_entropy(rho_dict):
    """calculate entanglement entropy from density matrices of two qubits

    Args:
        rho_dict (dictionary): dictionary of density matrices 

    Returns:
        list: list of dictionaries of entanglement entropy
    """
    if isinstance(rho_dict, dict):
        rho_dict = [rho_dict]
    nexp = len(rho_dict)
    S_list = []
    
    for i in range(nexp):
        S_dict = {}
        for edge, bns in rho_dict[i].items():
            S_total = 0
            for bn, rho in bns.items():
                rho_qiskit = DensityMatrix(rho)
                #calculate partial trace on first qubit (doesn't matter if it's on second)
                reduced_rho = partial_trace(rho_qiskit, [0]).data
                #find physical density matrix
                reduced_rho_physical = GraphState.find_closest_physical(reduced_rho)
                #find eigenvalues
                eigvals = np.linalg.eigvals(reduced_rho_physical)
                
                remove_idx = []
                for i in range(len(eigvals)):
                    if eigvals[i] == 0:
                        remove_idx.append(i)
                # record non-zero eigenvalues
                new_eigvals = np.delete(eigvals, remove_idx)
                #calculate the entropy
                S = -np.sum(new_eigvals*np.log2(new_eigvals))
                S_total += S
                
            S_mean = S/len(bns)#average entropy over all neighbour projections
            S_dict[edge] = S_mean
            
        S_list.append(S_dict)
    return S_list


def calc_n(rho):
    """Calculate the negativity of bipartite entanglement for a given 2-qubit
    density matrix

    Args:
        rho (numpy2d array): density matrix

    Returns:
        float>0: the negativity 0<n<0.5
    """
    rho_pt = ptrans(rho)#partial transpose 
    w, _ = la.eig(rho_pt)#eigenvalues
    n = np.sum(w[w < 0])#only sum the negative eigenvalues

    return abs(n)

def ptrans(rho):
    """Obtain the partial transpose of a 4x4 array (A kron B) w.r.t B

    Args:
        rho (numpy2d array 4*4): density matrix

    Returns:
        numpy2d array: 4*4 partial transposed density matrix (tranposing as if transposing one of 
        its tensor decomposition)
    """
    rho_pt = np.zeros(rho.shape, dtype=complex)
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            rho_pt[i:i+2, j:j+2] = rho[i:i+2, j:j+2].transpose()

    return rho_pt

def plot_negativities_multi(backend, n_list, nmit_list=None, figsize=(6.4, 4.8)):
    """Plot average negativity across multiple experiments with error bars as std

    Args:
        backend (IBMProvider().backend): IBM backend
        n_list (list): list of negativities in dict
        nmit_list (list, optional): list of mitigated negativities in dict. Defaults to None.
        figsize (tuple, optional): size of output figure. Defaults to (6.4, 4.8).

    Returns:
        fig: matplotlib figure of the plot of negativity vs edges
    """

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract the mean negativity and its standard deviation
    edges = n_list[0].keys()
    n_mean, n_std = calc_n_mean(n_list)

    # Convert into array for plotting
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in edges])
    Y0 = np.fromiter(n_mean.values(), float)
    Y0err = np.fromiter(n_std.values(), float)

    # If mitigated results are included
    try:
        nmit_mean, nmit_std = calc_n_mean(nmit_list)

        Y1 = np.fromiter(nmit_mean.values(), float)
        Y1err = np.fromiter(nmit_std.values(), float)
        # Order in increasing minimum negativity (QREM)
        Y1min = Y1 - Y1err
        idx = Y1min.argsort()#find the indicies that sort Y1min from smallest to largest
        Y1 = Y1[idx]#then put all negativities in such order (indicies)
        Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        idx = Y0min.argsort()

    X = X[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]

    # Plot
    ax.errorbar(X, Y0, yerr=Y0err, capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.4f})')
    try:
        ax.errorbar(X, Y1, yerr=Y1err, capsize=3, fmt='.', c='b', 
                    label=f'QREM (Mean negativity: {np.mean(Y1):.4f})')
    except:
        pass

    # Fig params
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    ax.legend()

    ax.set_xlabel("Qubit Pairs")
    ax.set_ylabel("Negativity")
    #ax.set_title(f"Native-graph state negativities ({backend.name()})")
    ax.set_title(backend.name())

    return fig


def plot_cxerr_corr(properties, adj_edges, n_mean, inc_adj=True, figsize=(6.4, 4.8)):
    """Plot negativity vs. CNOT error

    Args:
        properties (backend.properties()): details of gate errors/time on specific backend
        adj_edges (dictionary): dictionary of adjacent (indicent) edges to the current edges
        n_mean (dictionary): dictionary of negativities (mean over projects)
        inc_adj (bool, optional): whether to include adjacent edges. Defaults to True.
        figsize (tuple, optional): size of output figure. Defaults to (6.4, 4.8).

    Returns:
        two numpy arraies: CNOT errors and mean negativities on each edge
    """
    # Figure
    #fig, ax = plt.subplots(figsize=figsize)

    edges = n_mean.keys()
    
    if inc_adj is True:
        X = []
        for edge in edges:
            targ_err = [properties.gate_error('cx', edge)]
            adj_errs = [properties.gate_error('cx', adj_edge) 
                       for adj_edge in adj_edges[edge]
                       if adj_edge in edges] # list of CNOT errors for all adjacent edges of this particular edge
            err = np.mean(targ_err + adj_errs) #Take the error to be the average of current edge and its adjacents
            X.append(err)
        X = np.array(X)
    else:
        X = np.fromiter((properties.gate_error('cx', edge)
                         for edge in edges), float)

    Y = np.fromiter((n_mean.values()), float)

    #ax.scatter(X, Y)
    #ax.set_xlabel("CNOT Error")
    #ax.set_ylabel("Negativity")

    return X, Y

def calc_n_mean(n_list):
    """Calculate mean negativity dict from lists of negativity dicts

    Args:
        n_list (list): list of dictionaries of negativities

    Returns:
        two dictionaries: dictionary of mean negativities and standard errors
    """

    edges = n_list[0].keys()
    N = len(n_list)

    n_dict = {edge: [n_list[i][edge] for i in range(N)] for edge in edges}# n_dict= {'(0,1)':[0.4,0.3,0.44...],'(1,2)':[0.3,0.34,0.4...]}
                                                                          # which is the list of negativities obtained in different runs
                                                                          # corresponds to each pair of qubits
    n_mean = {key: np.mean(value) for key, value in n_dict.items()}#n_mean = {'(0,1)':0.44,'(1,2)':0.23} where the negativities are averaged
    n_std = {key: np.std(value)/np.sqrt(N) for key, value in n_dict.items()}#n_std are the standard deviation for each pair of qubits

    return n_mean, n_std


def plot_device_nbatches(provider, size=(6.4, 4.8)):
    """Plot the number of QST patches for each available device

    Args:
        provider (IBMProvider()): provider of the backend
        size (tuple, optional): figure output size. Defaults to (6.4, 4.8).

    Returns:
        fig: plot number of qubits vs number of batches can be generated in different IBM machines
    """
    # Figure
    fig, ax = plt.subplots(figsize=size)

    X = []  # Name
    Y = []  # No. of batches
    N = []  # No. of qubits

    for backend in provider.backends():
        try:
            properties = backend.properties()
            nqubits = len(properties.qubits)
            name = properties.backend_name

            nbatches = len(GraphState(backend).gen_batches())

            X.append(name + f', {nqubits}')
            Y.append(nbatches)
            N.append(nqubits)

        except:
            pass

    # Convert to numpy arrays for sorting
    X = np.array(X)
    Y = np.array(Y)
    N = np.array(N)
    # Sort by number of qubits
    idx = N.argsort()
    X = X[idx]
    Y = Y[idx]

    # Plot
    ax.scatter(X, Y)
    ax.tick_params(axis='x', labelrotation=90)

    return fig
