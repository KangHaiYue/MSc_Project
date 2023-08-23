# -*- coding: utf-8 -*-
"""
Last Updated on 19 August, 2023
@author: Haiyue Kang
"""

# Standard libraries
import queue as Q
import copy
#other installed libraries
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import numpy as np
import numpy.linalg as la
import networkx as nx
# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute, transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import InstructionDurations
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix, Operator, PauliList
from qiskit_ibm_provider import IBMProvider
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
#I tensor H operator in matrix
H1 = np.kron(np.array([[1, 0],[0, 1]]), 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]))
#I tensor X operator in matrix
X1 = np.kron(np.array([[1, 0],[0, 1]]), np.array([[0, 1], [1, 0]]))
#Base Bell State
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

States_odd = []
States_even = []
for circuit in BS_list_odd:
    state = Statevector(circuit)
    States_odd.append(state)
for circuit in BS_list_even:
    state = Statevector(circuit)
    States_even.append(state)
#Direct Matrix form of the 8 variants (4 variants up to Hadamard)
BS_list_odd[0] = 1/np.sqrt(2)*np.array([[1],[0],[0],[1]])
BS_list_odd[1] = 1/np.sqrt(2)*np.array([[0],[1],[1],[0]])
BS_list_odd[2] = 1/np.sqrt(2)*np.array([[1],[0],[0],[-1]])
BS_list_odd[3] = 1/np.sqrt(2)*np.array([[0],[1],[-1],[0]])
BS_list_even[0] = 1/2*np.array([[1],[1],[1],[-1]])
BS_list_even[1] = 1/2*np.array([[1],[1],[-1],[1]])
BS_list_even[2] = 1/2*np.array([[1],[-1],[1],[1]])
BS_list_even[3] = 1/2*np.array([[-1],[1],[1],[1]])

def find_closest_pvec(pvec_array):
    """Find closest probability vector
    [C Michelot - Journal of Optimization Theory and Applications, 1986]
    works for both prob. vectors and shot count vectors
    Args:
        pvec_array (numpy array of dimension 2^N): probability vector, in principle non-physical
        (may have negative values)

    Returns:
        numpy array of same size as input: corrected physical probability vector
    """
    # placeholder vector
    v = lil_matrix(pvec_array)
    q = Q.PriorityQueue()

    cv = v.tocoo()
    #cv = vector_as_sparse_matrix.tocoo()
    count = 0

    for i,j,k in zip(cv.row, cv.col, cv.data):
        q.put((k, (i,j)))
        count += 1
    # q now stores eigenvalues in increasing order with corresponding index as value.
    # note that we don't need to have count = vector_as_sparse_matrix.shape[0] because count is not important for
    # negative values (since (item[0] + a / count) < 0 is always true) and after processing the negative and zero 
    # values count would be decremented by #(zero values) anyway. So we can ignore zero values when calculating count

    a = 0
    continue_zeroing = True
    while not q.empty():
        item = q.get()
        if continue_zeroing:
            if (count > 0) and ((item[0] + a / count) < 0):
                v[item[1][0], item[1][1]] = 0
                a += item[0] # add up all the pvec elements xi that has xi+a/count < 0
                count -= 1 #eventually count will be reduced to number of elements left in the pvec that has xi +a/count > 0
            else:
                continue_zeroing = False #once the judgement fails, it never get back to True
        if not continue_zeroing:
            v[item[1][0], item[1][1]] = item[0] + a / count #update the rest of the pvec items by distributing the taking away the -ves 
            #that has been set to 0


    return v.toarray()[0]


class Teleportation(Free_EntangleBase):
    """
    Class to run and analyse two-qubit graph state teleportation hopping along a chain of qubits
    """

    def __init__(self, backend, qubits_to_connect):
        """
        Initialize from Free_EntangleBase parent class and additionally obtain
        edge adjacencies and generate the native-graph state preperation
        circuit

        Args:
            backend (IBMProvider().backend): IBM quantum computer backend
            qubits_to_connect (list): list of qubits to connect
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
        
        durations = InstructionDurations.from_backend(backend)
        self.tx = durations.get('x', 0)
        self.tz = durations.get('rz', 0)
        
        self.teleportation_basis = None
        self.teleported_BellState_circuits = None
        self.teleported_BellState_circuits_qst = None
        

    def __get_adjs(self):
        """
        Get the edge-qubit adjacencies for every physical edge in the device.
        Keys are edges (tuple) and values are adjacent qubits (list)

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

    def gen_graphstate_circuit(self, return_depths=False):
        """
        Generate a native-graph state circuit over every physical edge. Note that the CZ gates have their maximum depths always = 3
        (see Fidel's thesis Fig 3.2)
        Args:
            return_depths (bool, optional): whether return the actual circuit or circuit depth. Defaults to False.

        Returns:
            Qiskit-QuantumCircuit: circuit of the graphstate preparation
        """
        circ = QuantumCircuit(self.device_size)
        unconnected_edges = self.edge_list.copy()
        depths = []
        # Apply Hadamard gates to every qubit
        circ.h(self.qubits_to_connect)
        # Connect every edge with cz gates
        while unconnected_edges:
            connected_qubits = []  # Qubits already connected in the current time step
            remove = []
            for edge in unconnected_edges:
                if np.any(np.isin(edge, connected_qubits)) == False:
                    circ.cz(edge[0], edge[1])
                    connected_qubits.extend(edge)
                    remove.append(edge)
            # Remove connected edges from unconnected edges list
            depths.append(remove)
            for edge in remove:
                unconnected_edges.remove(edge)
                
        if return_depths is True:
            return depths

        return circ
    
    def gen_delay_circuit(self, t, increment, dynamic_decoupling=False):
        """(Deprecated) Generate delay circuits based on the graphstate circuit and delay option

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
        """(Deprecated) Generate periodic dynamical decoupling with 2 pulses in each increment

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
        """(Deprecated) Generate double X pulse dynamical decoupling circuit

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
        """(Deprecated) Generate single X pulse dynamical decoupling circuit

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
        """format the delays times in the unit of qiskit delay gate qunta

        Args:
            delays (list): list of delay times
            unit (str, optional): unit of delays input. Defaults to 'ns'.

        Returns:
            numpy list: formatted delay
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
    

    def apply_mthree(self, pvecs_list, counts_list, mit=None):
        """apply M3 mitigation to counts/pvecs

        Args:
            pvecs_list (list): list of probability vectors dictionaries
            counts_list (list): list of counts dictionaries
            mit : mthreeMitigation object with corresponding backend Defaults to None.

        Returns:
            two lists of dictionaries: mitigated pvecs and counts
        """
        if mit is None:
            mit = load_cal(self.backend)
        self.mit = mit
        
        counts_list_mit = counts_list.copy()
        pvecs_list_mit = pvecs_list.copy()
        
        for i in range(self.reps):
            for basis, counts in counts_list[i].items():
                corrected_counts = self.mit.apply_correction(counts, list(range(self.nqubits)))
                counts_list_mit[i][basis] = corrected_counts.nearest_probability_distribution()
                #Unnormalise
                for bit_str, prob in counts_list_mit[i][basis].items():
                    counts_list_mit[i][basis][bit_str] = prob*self.shots
                    bit_str_binary = bit_str[::-1]
                    idx = int(bit_str_binary, 2)
                    pvecs_list_mit[i][basis][idx] = prob
        
        return pvecs_list_mit, counts_list_mit
    

        
    def run_qst(self, reps=1, shots=4096, qrem=False, sim=None, output='default',
                execute_only=False):
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
        job = self.run_qst_circuits(reps, shots, qrem, sim)

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
        """ (Deprecated)
        Get a dictionary of tomography batches, where keys are batch numbers
        and values are lists of tomography groups (targets + adj qubits).
        QST can be performed on batches in parallel.

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
    
    def gen_teleportation_basis(self):
        """get a dictionary of measurement basis for each qubit, where key is the qubit pair want in Bell state,
        with sub-dctionary specifying the X and Z measurement qubits indicies

        Returns:
            dictionary: measurement basis for teleportation
        """
        #Find the possible qubit-pair combinations
        #{1:{(0,3):{'X':[1,2],'Z':},(1,3):{'X'[],'Z':[]}}, 2:{...}}
        possible_pairs = [(i,j) for i in self.qubits_to_connect for j in self.qubits_to_connect if i < j]
        largest_gap = 0
        #Find the largest gap possible in this topolgy of the quantum computer
        for pair in possible_pairs:
            if nx.shortest_path_length(self.graph, pair[0], pair[1], weight = 1) > largest_gap:
                #Note in fact the so called largest_gap is actually the 'greatest number of steps from one end to other' 
                #so it overcounts by 1
                largest_gap = nx.shortest_path_length(self.graph, pair[0], pair[1], weight = 1)
         
        
        teleportation_basis = {gap:{} for gap in range(1, largest_gap)}
        
        for gap in range(1, largest_gap):#since largest_gap overcounts by 1 the range method automatically cancels
            #unused pair of qubits of gap = gap
            unused_pairs = [(i,j) for i in self.qubits_to_connect for j in self.qubits_to_connect 
                            if nx.shortest_path_length(self.graph, i, j) == gap + 1 and i < j]
            basis_dict = {pair:{} for pair in unused_pairs}
            #find the qubit indicies of the X measurement and Z measurements
            for pair in unused_pairs:
                basis_dict[pair]['X'] = nx.shortest_path(self.graph, pair[0], pair[1])[1:-1] #not taking the last qubit
                all_nodes = list(nx.nodes(self.graph))
                Z_measurements = [qubit for qubit in all_nodes if qubit not in nx.shortest_path(self.graph, pair[0], pair[1])]
                basis_dict[pair]['Z'] = Z_measurements
                
            teleportation_basis[gap] = basis_dict
        self.teleportation_basis = teleportation_basis
        return teleportation_basis
    
    def gen_chain_graphstate_circuit(self, qubit_pair, X_basis_qubits):
        """Generate a graph state on a graph of a path (or chain), ends are qubit_pair

        Args:
            qubit_pair (list or tuple): ending qubits of the chain
            X_basis_qubits (list): list of intermediate qubits

        Returns:
            qiskit QuantumCircuit: circuit to generate the chain graph state, depth is always 3
        """
        circ = QuantumCircuit(self.device_size)
        connection_order = [list(qubit_pair)[0]] + X_basis_qubits + [list(qubit_pair)[1]]

        # Apply Hadamard gates to every qubit
        circ.h(connection_order)
        # Connect every edge with cz gates
        for i in range(0,len(connection_order)-1,2):
            circ.cz(connection_order[i], connection_order[i+1])
        for i in range(1,len(connection_order)-1,2):
            circ.cz(connection_order[i], connection_order[i+1])

        return circ
    
    def gen_two_qubit_graphstate_circuit(self, qubit_a, qubit_b):
        """Generate a graph state with only 2 qubits (nearest neighbour)

        Args:
            qubit_a (int): index of qubit a
            qubit_b (int): index of qubit b

        Returns:
            qiskit QuantumCircuit: circuit to generate the two-qubit graph state, depth is 2
        """

        circ = QuantumCircuit(self.device_size)
        connection_order = [qubit_a] + [qubit_b]

        # Apply Hadamard gates to every qubit
        circ.h(connection_order)
        # Connect every edge with cz gates
        circ.cz(qubit_a, qubit_b)

        return circ
    
    def gen_teleported_BellState_circuit(self, post_processing = True):
        """Generate Teleportation circuit, no QST yet

        Args:
            post_processing (bool, optional): _description_. Defaults to True.

        Returns:
            dictionary: dictionary of teleportation circuit
        """
        #Make sure the measurement basis for each pair of qubits (and intermediate qubits) are found
        self.post_processing = post_processing
        if self.teleportation_basis is None:
            self.teleportation_basis = self.gen_teleportation_basis()
        
        BellState_circuits = {}  
        
        #graphstate = self.circuit.copy()
        #graphstate.barrier()
        
        for gap, pairs_dict in self.teleportation_basis.items():
            gap_circuits = {}
            for qubit_pair, basis_dict in pairs_dict.items():
                X_basis_qubits = basis_dict['X']
                #generate graph state circuit for each pair of qubits
                graphstate = self.gen_chain_graphstate_circuit(qubit_pair, X_basis_qubits)
                graphstate.barrier()
                
                circ = graphstate.copy(f'{gap}-{qubit_pair}')
                crX = ClassicalRegister(gap)
                circ.add_register(crX)
                
                #circ.barrier()
                #measure intermediate qubits in X basis
                circ.h(np.array(X_basis_qubits).flatten())
                circ.measure(np.array(X_basis_qubits).flatten(), crX)
                #Add dynamic circuit if not post processed/categorised circuit
                if post_processing is False:
                    for i in range(len(X_basis_qubits))[::-1]:
                        with circ.if_test((crX[i],1)):
                            circ.x(qubit_pair[1])
                        #circ.x(qubit_pair[1]).c_if(crX[i],1)
                        circ.h(qubit_pair[1])
                gap_circuits[qubit_pair] = circ
             
            BellState_circuits[gap] = gap_circuits
            
        self.teleported_BellState_circuits = BellState_circuits
        return BellState_circuits
    
    def gen_swap_BellState_circuit(self):
        """generate BellState transferring circuit (moving one qubit away by SWAP gates)

        Returns:
            dictionary: dictionary of circuits that use SWAP gates to move information in a qubit
        """
        #Make sure the measurement basis for each qubit pair (and intermediate qubits) are found
        self.post_processing = True
        if self.teleportation_basis is None:
            self.teleportation_basis = self.gen_teleportation_basis()
            
        BellState_circuits = {}  
        for gap, pairs_dict in self.teleportation_basis.items():
            gap_circuits = {}
            for qubit_pair, basis_dict in pairs_dict.items():
                X_basis_qubits = basis_dict['X']
                #generate two-qubit graph state in nearest neighbour
                graphstate = self.gen_two_qubit_graphstate_circuit(list(qubit_pair)[0], X_basis_qubits[0])
                graphstate.barrier()
                
                circ = graphstate.copy(f'{gap}-{qubit_pair}')
                swap_controls = X_basis_qubits
                swap_targets = X_basis_qubits[1:] + [list(qubit_pair)[1]]
                #move the information in the second qubit away using SWAP gates
                for i in range(len(swap_controls)):
                    circ.swap(swap_controls[i], swap_targets[i])
                
                gap_circuits[qubit_pair] = circ
             
            BellState_circuits[gap] = gap_circuits
            
        self.teleported_BellState_circuits = BellState_circuits
        return BellState_circuits
    
    def gen_teleported_qst_circuits(self, teleportation = 'teleportation'):
        """Generate QST circuits after Teleportation

        Args:
            teleportation (str, optional): mode of teleportation or swap. Defaults to 'teleportation'.

        Returns:
            dictionary: dictionary of circuits in gap-qubit_piar-basis structure
        """
        #generate the teleportation circuit according to the mode wanted
        if self.teleported_BellState_circuits is None:
            if teleportation == 'teleportation':
                self.teleported_BellState_circuits = self.gen_teleported_BellState_circuit(self.post_processing)
            if teleportation == 'swap':
                self.teleported_BellState_circuits = self.gen_swap_BellState_circuit()
        #set up the dictionary for storing circuits
        qst_circuits = {gap: {qubit_pair:{} for qubit_pair in pairs_dict} 
                        for gap, pairs_dict in self.teleported_BellState_circuits.items()}
        name_list = []
        
        for gap, pairs_dict in qst_circuits.items():
            for qubit_pair in pairs_dict.keys():
                targets = np.array(qubit_pair).flatten()
                #change the measurement basis to desired pauli-operators
                circxx = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-XX')
                circxx.h(targets)
                qst_circuits[gap][qubit_pair]['XX'] = circxx
                
                circxy = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-XY')
                circxy.sdg(targets[1])
                circxy.h(targets)
                qst_circuits[gap][qubit_pair]['XY'] = circxy
                
                circxz = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-XZ')
                circxz.h(targets[0])
                qst_circuits[gap][qubit_pair]['XZ'] = circxz
                
                circyx = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-YX')
                circyx.sdg(targets[0])
                circyx.h(targets)
                qst_circuits[gap][qubit_pair]['YX'] = circyx
                
                circyy = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-YY')
                circyy.sdg(targets)
                circyy.h(targets)
                qst_circuits[gap][qubit_pair]['YY'] = circyy
                
                circyz = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-YZ')
                circyz.sdg(targets[0])
                circyz.h(targets[0])
                qst_circuits[gap][qubit_pair]['YZ'] = circyz
                
                circzx = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-ZX')
                circzx.h(targets[1])
                qst_circuits[gap][qubit_pair]['ZX'] = circzx
                
                circzy = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-ZY')
                circzy.sdg(targets[1])
                circzy.h(targets[1])
                qst_circuits[gap][qubit_pair]['ZY'] = circzy
                
                circzz = self.teleported_BellState_circuits[gap][qubit_pair].copy(f'{gap}-{qubit_pair}-ZZ')
                qst_circuits[gap][qubit_pair]['ZZ'] = circzz
                #another measurement circuit for QST (distinct from teleportation measurment)
                for circ in qst_circuits[gap][qubit_pair].values():
                    name_list.append(circ.name)
                    cr3 = ClassicalRegister(2)
                    circ.add_register(cr3)
                    circ.measure(targets, cr3)
        
        self.teleported_BellState_circuits_qst = qst_circuits
        self.name_list = name_list
        return qst_circuits
        
    def run_teleported_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None):
        """submit the job to IBMBackend

        Args:
            reps (int, optional): total number of repetitions on circuit list. Defaults to 1.
            shots (int, optional): number of shots per circuit. Defaults to 4096.
            qrem (bool, optional): whether add QREM circuit to run. Defaults to False.
            sim (_type_, optional): simulator mode. Defaults to None.

        Returns:
            IBMJob: job submitted to run
        """
        self.reps = reps
        self.shots = shots
        self.qrem = qrem
        self.sim = sim
        # Convert circuits dict into list form
        circ_list = []
        for pairs_dict in self.teleported_BellState_circuits_qst.values():
            for basis_dict in pairs_dict.values():
                for circuit in basis_dict.values():
                    circ_list.append(circuit)
        # Extend circuit list by number of repetitions            
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        
        circ_list = circ_list_multi
        # Generate QREM circuits and append to circ_list if qrem == True
        job_qrem = None
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            # If circuits are executed on a simulator or real backend or runtime sampler mode
            if sim is None:
                qrem_transpiled = transpile(qrem_circuits, backend = self.backend)
                job_qrem = self.backend.run(qrem_transpiled, shots=shots)
                #job_qrem = execute(qrem_transpiled, backend = self.backend, shots=shots)
            elif sim == 'ideal':
                backend = Aer.get_backend('aer_simulator')
                job_qrem = execute(qrem_circuits, backend=backend, 
                              initial_layout=list(range(self.device_size)),
                              shots=shots)
            elif sim == 'device':
                # Obtain device and noise model parameters
                noise_model = NoiseModel.from_backend(self.backend)
                coupling_map = self.backend.configuration().coupling_map
                basis_gates = noise_model.basis_gates
                backend = Aer.get_backend('aer_simulator')
            
                job_qrem = execute(qrem_circuits, backend=backend,
                              coupling_map=coupling_map,
                              basis_gates=basis_gates,
                              noise_model=noise_model,
                              shots=shots)
        #run dynamic circuit
        dynamic = not self.post_processing
        if dynamic is True:
            #run in according simulator mode
            if sim is None:
                circ_list_transpiled = transpile(circ_list, backend = self.backend)
                #job = execute(circ_list, backend=self.backend, shots=shots, dynamic = dynamic)
                job = self.backend.run(circ_list_transpiled, shots=shots, dynamic=dynamic)
                
            elif sim == "ideal":
                backend = Aer.get_backend('aer_simulator')
                #job = execute(circ_list, backend=backend, 
                #              initial_layout=list(range(self.nqubits)),
                #              shots=shots, dynamic = dynamic)
            
                circ_list_transpiled = transpile(circ_list, backend = backend,
                                                 initial_layout=list(range(self.device_size)))
                job = backend.run(circ_list_transpiled, shots=shots, dynamic=dynamic)
            elif sim == "device":
                # Obtain device and noise model parameters
                noise_model = NoiseModel.from_backend(self.backend)
                coupling_map = self.backend.configuration().coupling_map
                basis_gates = noise_model.basis_gates
                backend = Aer.get_backend('aer_simulator')
                circ_list_transpiled = transpile(circ_list, backend = backend,
                                                 initial_layout=list(range(self.device_size)))
                job = backend.run(circ_list_transpiled,
                                  coupling_map=coupling_map,
                                  basis_gates=basis_gates,
                                  noise_model=noise_model,
                                  shots=shots, dynamic = dynamic)
        # if no qrem nor dynamic circuit
        else:
            if sim is None:
                circ_list_transpiled = transpile(circ_list, backend = self.backend)
                job = self.backend.run(circ_list_transpiled, shots=shots)
                #job = execute(circ_list, backend=self.backend, shots=shots)
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
            
        return job, job_qrem
    
    def teleported_counts_from_result(self, result, result_qrem, post_processing = True,
                                      teleportation = 'teleportation'):
        """Get counts from teleportation from qiskit result as dictionary or lists of dictionaries

        Args:
            result (IBMJob.result()): result completed
            result_qrem (IBMJob.result()): QREM result completed
            post_processing (bool, optional): whether circuit was designed for post-process. Defaults to True.
            teleportation (str, optional): mode of teleportation. Defaults to 'teleportation'.

        Returns:
            two lists: list of counts and probability vectors
        """
        #Initialize all prerequired self objects from result
        self.post_processing = post_processing
        if self.name_list is None:
            self.gen_teleported_qst_circuits(teleportation = teleportation)
        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))# number of qubits？/total number of QST circuits need to run 
                                                                    #s.t. all pairs are fulled QST
            self.shots = result.results[0].shots
        try:  # Try to obtain QREM results
            result_qrem.get_counts('qrem0')
            self.qrem = True
        except:
            self.qrem = False

        # Load counts as dict experiment-wise
        qst_counts_multi = []
        pvecs_multi = []
        for i in range(self.reps):
            qst_counts = copy.deepcopy(self.teleported_BellState_circuits_qst)
            pvecs = copy.deepcopy(self.teleported_BellState_circuits_qst)
            #use name_list defined when genearte the circuit, construct the dictionary
            for name in self.name_list:
                gap, qubit_pair, basis = name.split('-')
                q_left, q_right = qubit_pair.split(',')
                qubit_pair_formatted = (int(q_left[1:]),int(q_right[:-1]))
                name_ext = name + f'-{i}'
                qst_counts[int(gap)][qubit_pair_formatted][basis] = result.get_counts(name_ext)
                #qst_counts[2][(1,4)]['ZZ']={'00 00':21,'00 01':12,....}
                #storing information differently according to teleportation mode
                if teleportation == 'teleportation':
                    pvecs[int(gap)][qubit_pair_formatted][basis] = np.zeros(2**(int(gap)+2))
                    for bit_str, count in result.get_counts(name_ext).items():
                        pair_str, X_str = bit_str.split(' ')
                        new_str = pair_str + X_str
                        idx = int(new_str[::-1], 2)
                        pvecs[int(gap)][qubit_pair_formatted][basis][idx] += count
                        
                elif teleportation == 'swap':
                    pvecs[int(gap)][qubit_pair_formatted][basis] = np.zeros(4)
                    for bit_str, count in result.get_counts(name_ext).items():
                        idx = int(bit_str[::-1], 2)
                        pvecs[int(gap)][qubit_pair_formatted][basis][idx] += count
                pvecs[int(gap)][qubit_pair_formatted][basis] /= self.shots
                
            qst_counts_multi.append(qst_counts) #e.g. qst_counts_multi = [{qst_counts 1st iter}, {qst_counts 2nd iter},...]
            pvecs_multi.append(pvecs)

        # Save list of calibration matrices for each qubit
        if self.qrem is True:
            qrem_counts = [result_qrem.get_counts('qrem0'),
                           result_qrem.get_counts('qrem1')]

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

        return qst_counts_multi, pvecs_multi  # Multiple experiments
    
    def teleported_counts_from_result_list(self, result_dict, qrem_dict=None):
        """only for dynamic circuit results dict, results dict are read from csv file: key are rep-gap-qubit_pair, value are job ids

        Args:
            result_dict (dictionary): dictionary of results, key tells rep-gap-qubit_pair and values are the job_id
            qrem_dict (dictionary, optional): dictionary of QREM results, key tells rep-gap-qubit_pair and values are QREM job_id. 
            Defaults to None.

        Returns:
            list: list of counts and probability vectors
        """
        #generate all prerequired basis and namelist
        self.post_processing = False
        if self.name_list is None:
            self.gen_teleported_qst_circuits()
        provider = IBMProvider()
        
        if self.reps is None:
            self.reps = int(len(result_dict)*9/len(self.name_list))
        
        qst_counts_multi = [copy.deepcopy(self.teleported_BellState_circuits_qst) for i in range(self.reps)]
        pvecs_multi = [copy.deepcopy(self.teleported_BellState_circuits_qst) for i in range(self.reps)]
        #store the results into counts/pvecs
        for job_name, job_id in result_dict.items():
            rep, gap, qubit_pair = job_name.split('-')
            q_left, q_right = qubit_pair.split(',')
            qubit_pair_formatted = (int(q_left[1:]),int(q_right[:-1]))

            job = provider.backend.retrieve_job(job_id)
            result = job.result()
            self.shots = result.results[0].shots
            # get the counts from the name recovered
            for basis in basis_list:
                name_ext = gap + '-' + qubit_pair + '-' + basis + '-' + rep
                qst_counts_multi[int(rep)][int(gap)][qubit_pair_formatted][basis] = result.get_counts(name_ext)
                pvecs_multi[int(rep)][int(gap)][qubit_pair_formatted][basis] = np.zeros(2**(2+int(gap)))
                #store pvec index according to bit-strings
                for bit_str, count in result.get_counts(name_ext).items():
                    pair_str, X_str = bit_str.split(' ')
                    new_str = pair_str + X_str
                    idx = int(new_str[::-1], 2)
                    pvecs_multi[int(rep)][int(gap)][qubit_pair_formatted][basis][idx] += count
                pvecs_multi[int(rep)][int(gap)][qubit_pair_formatted][basis] /= self.shots
        
        #recover QREM results/ calirbation matrices
        if qrem_dict is not None:
            self.qrem = True
            result = provider.backend.retrieve_job(qrem_dict['qrem']).result()
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
        else:
            self.qrem = False

        return qst_counts_multi, pvecs_multi  # Multiple experiments
    
    def apply_reduced_qrem_to_pvec(self, pvec, gap, measurements_order, mitigate_qubits = [1,3,4,5]):
        """Apply QREM qubit-wisely on selected qubits

        Args:
            pvec (numpy array): probability vector
            gap (int): number of hops in teleportation
            measurements_order (_type_): the indicies of qubits in the same order of their measurement order
            mitigate_qubits (list, optional): list of qubits to mitigate. Defaults to [1,3,4,5].

        Returns:
            numpy array: corrected probability vector
        """
        mitigate_qubits_true = list(set(mitigate_qubits).intersection(measurements_order))
        threshold = 0.001/self.shots#set minimum threshold, zero out the count if below it
        #iterate over each qubit
        for q in mitigate_qubits_true:
            idx = measurements_order.index(q)
            calibration_M = la.inv(self.M_list[q])
            applied_names = set([])
            bit_string_ints = np.flatnonzero(pvec.copy())#non-zero indicies of the pvec elements
            for num in bit_string_ints:
                bit_string = bin(num)[2:].zfill(gap+2)
                bit_string_list = list(bit_string)
                bit_string_list[idx] = '_'
                name = "".join(bit_string_list)
                #check if the bit-string (execpt digit q) is already been corrected
                if name not in applied_names:
                    applied_names.add(name)
                    #check the digit is 0 or 1, then flip it
                    if (num & (1 << idx)) != 0:
                        bit_string_list[idx] = '0'
                    else:
                        bit_string_list[idx] = '1'
                    bit_string_flip = "".join(bit_string_list)
                    num_flip = int(bit_string_flip, 2)
                    
                    reduced_pvec = np.zeros(2)
                    # if 0->1
                    if num < num_flip:
                        reduced_pvec[0] += pvec[num]
                        reduced_pvec[1] += pvec[num_flip]
                        #calibrate the two elemnts with bit-string only differ at this digit idx
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                        if abs(reduced_pvec_mit[0]) > threshold:
                            pvec[num] = reduced_pvec_mit[0]
                        else:
                            pvec[num] = 0
                        if abs(reduced_pvec_mit[1]) > threshold:
                            pvec[num_flip] = reduced_pvec_mit[1]
                        else:
                            pvec[num_flip] = 0
                    #if 1->0          
                    else:
                        reduced_pvec[1] += pvec[num]
                        reduced_pvec[0] += pvec[num_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            pvec[num_flip] = reduced_pvec_mit[0]
                        else:
                            pvec[num_flip] = 0
                        if abs(reduced_pvec_mit[1]) > threshold:
                            pvec[num] = reduced_pvec_mit[1]
                        else:
                            pvec[num] = 0
        pvec = find_closest_pvec(pvec)
        print(f'{np.count_nonzero(pvec)}')
        return pvec
        
    def apply_qrem_teleported_counts(self, qst_counts, pvecs, teleportation = 'teleportation', 
                                     qrem='QREM', mitigate_qubits = [1,3,4,5]):
        """Apply QREM

        Args:
            qst_counts (list): list of counts dictionaries
            pvecs (list): list of pvec dictionaries
            teleportation (str, optional): teleportation mode. Defaults to 'teleportation'.
            qrem (str, optional): error mitigation mode. Defaults to 'QREM'.
            mitigate_qubits (list, optional): qubits to mitigate if using reduced_QREM. Defaults to [1,3,4,5].

        Returns:
            two lists: corrected qst_counts and pvecs
        """
        qst_counts_list_mit = [] #pvecs=[{1:{(1,2):{'XX':[probability vec],'XY':...},(2,3):{'XX':...,'XY':...},(3,4):...}, 2:{...}}, {...}]
        pvecs_list_mit = []
        for i in range(self.reps):
            qst_counts_mit = copy.deepcopy(qst_counts[i])
            pvecs_mit = copy.deepcopy(pvecs[i])
            for gap, pairs_dict in pvecs[i].items():
                for pair, basis_dict in pairs_dict.items():
                    #check the mode of teleportation
                    if teleportation == 'teleportation':
                        X_measurements = self.teleportation_basis[gap][pair]['X']
                        targets = list(pair)
                        measurements_order = X_measurements + targets
                        # Invert n-qubit calibration matrix
                        if qrem == 'QREM':
                            M_inv = la.inv(self.calc_M_multi(measurements_order))
                        for basis, pvec in basis_dict.items():
                            #correct pvec according to mitigation mode
                            if qrem == 'QREM':
                                pvec_mit = np.matmul(M_inv, pvec)
                            elif qrem =='reduced_QREM':
                                pvec_mit = self.apply_reduced_qrem_to_pvec(pvec, int(gap), measurements_order,
                                                                           mitigate_qubits=mitigate_qubits)
                            #pvec_mit_physical = find_closest_pvec(pvec_mit)
                            pvecs_mit[gap][pair][basis] = pvec_mit
                            for j, prob in enumerate(pvec_mit):
                                bit_str = bin(j)[2:].zfill(gap+2)[::-1]
                                target_str = bit_str[:2]
                                X_str = bit_str[2:]
                                #Z_str = bit_str[2+len(X_measurements):]
                                bit_str_new = target_str + ' ' + X_str
                                qst_counts_mit[gap][pair][basis][bit_str_new] = prob*self.shots
                                
                    elif teleportation == 'swap':
                        measurements_order = list(pair)
                        # Invert n-qubit calibration matrix
                        M_inv = la.inv(self.calc_M_multi(measurements_order))
                        for basis, pvec in basis_dict.items():
                            pvec_mit = np.matmul(M_inv, pvec)
                            #pvec_mit_physical = find_closest_pvec(pvec_mit)
                            pvecs_mit[gap][pair][basis] = pvec_mit
                            for j, prob in enumerate(pvec_mit):
                                bit_str = bin(j)[2:].zfill(2)[::-1]
                                qst_counts_mit[gap][pair][basis][bit_str] = prob*self.shots
                        
            qst_counts_list_mit.append(qst_counts_mit)
            pvecs_list_mit.append(pvecs_mit)

        return qst_counts_list_mit, pvecs_list_mit
    
    def apply_qrem_teleported_counts_from_dynamic_circuits(self, qst_counts, pvecs):
        """Apply full QREM to dynamic circuits only

        Args:
            qst_counts (list): list of counts dictionaries
            pvecs (list): list of pvec dictionaries

        Returns:
            two lists: corrected counts/pvec dictionaries
        """
        qst_counts_list_mit = [] #pvecs=[{1:{(1,2):{'XX':[probability vec],'XY':...},(2,3):{'XX':...,'XY':...},(3,4):...}, 2:{...}}, {...}]
        pvecs_list_mit = []
        for i in range(self.reps):
            qst_counts_mit = copy.deepcopy(qst_counts[i])
            pvecs_mit = copy.deepcopy(pvecs[i])
            for gap, pairs_dict in pvecs[i].items():
                for pair, basis_dict in pairs_dict.items():
                    targets = list(pair)
                    # measurements_order = X_measurements + targets
                    # Invert n-qubit calibration matrix
                    M = np.identity(2**gap)
                    for target in targets:
                        M_new = np.kron(M, self.M_list[target])
                        M = M_new
                    M_inv = la.inv(M)
                    
                    for basis, pvec in basis_dict.items():
                        pvec_mit = np.matmul(M_inv, pvec)
                        #pvec_mit_physical = find_closest_pvec(pvec_mit)
                        pvecs_mit[gap][pair][basis] = pvec_mit
                        #store calibrated pvec into counts, separating the ending qubits and intermediate qubits
                        #bit strings
                        for j, prob in enumerate(pvec_mit):
                            bit_str = bin(j)[2:].zfill(gap+2)[::-1]
                            target_str = bit_str[:2]
                            X_str = bit_str[2:]
                            bit_str_new = target_str + ' ' + X_str
                            qst_counts_mit[gap][pair][basis][bit_str_new] = prob*self.shots
                        
                        
            qst_counts_list_mit.append(qst_counts_mit)
            pvecs_list_mit.append(pvecs_mit)

        return qst_counts_list_mit, pvecs_list_mit
    
    def bin_teleported_pvecs(self,qst_counts_list, pvecs_list):    
        """bin pvces list from gap-pair-QST basis-bit string(X-measurements+pair measurements) structure to gap-pair-BellState(X 
        measurements)-QST basis structure

        Args:
            qst_counts_list (list): list of counts dictionaries
            pvecs_list (list): list of pvecs dictionaries

        Returns:
            list: binned list of pvecs dictionaries
        """
        # bin the pvecs into corresponding Bell State
        pvecs_binned_list = []
        BellState_names = ['BS_1', 'BS_2', 'BS_3', 'BS_4']
        for i in range(self.reps):
            pvecs_binned = {}
            for gap, pairs_dict in qst_counts_list[i].items():
                pvecs_binned[gap] = {pair:{bellstate:{basis: np.zeros(4) for basis in basis_list}
                                            for bellstate in BellState_names}
                                        for pair in pairs_dict.keys()}
                
                for pair, basis_dict in pairs_dict.items():
                    for basis, counts in basis_dict.items():
                        for bit_str, count in counts.items():
                            #separate intermediate qubits and ending qubits
                            pair_str, X_str = bit_str.split(' ')
                            idx = int(pair_str[::-1], 2)
                            BellState_name = Teleportation.bit_str_to_BellState(X_str)#determine BS name from intermediate qubits
                            pvecs_binned[gap][pair][BellState_name][basis][idx] += count
                        
                        #Normalise each prob vector
                        for bellstate in BellState_names:
                            pvec = pvecs_binned[gap][pair][bellstate][basis]
                            if pvec.sum() != 0:
                                norm = 1/pvec.sum()
                                pvecs_binned[gap][pair][bellstate][basis] = pvec*norm
            pvecs_binned_list.append(pvecs_binned)
        
        return pvecs_binned_list
    
    def bin_teleported_pvecs_from_dynamic_circuits(self, qst_counts_list, pvecs_list):
        """truncate the pvecs list by combining all probabilities/counts which have same pair_str together (forget Z and X measures as 
        dynamic circuit already handled them

        Args:
            qst_counts_list (list): list of counts dictionaries
            pvecs_list (list): list of pvecs dictionaries

        Returns:
            list: binned list of pvecs dictionaries
        """
        pvecs_list_new = []
        for i in range(self.reps):
            pvecs_new = copy.deepcopy(pvecs_list[i])
            for gap, pairs_dict in qst_counts_list[i].items():
                for pair, basis_dict in pairs_dict.items():
                    for basis, counts in basis_dict.items():
                        pvecs_new[gap][pair][basis] = np.zeros(4)
                        for bit_str, count in counts.items():
                            pair_str = bit_str.split(' ')[0]
                            idx = int(pair_str[::-1], 2)
                            pvecs_new[gap][pair][basis][idx] += count
                        pvecs_new[gap][pair][basis] /= self.shots
                        
            pvecs_list_new.append(pvecs_new)
        return pvecs_list_new
        
        
    def recon_teleported_density_mats(self, result, result_qrem, post_processing = True, apply_mit='QREM',
                                      teleportation = 'teleportation', mitigate_qubits = [1,3,4,5]):
        """Reconstruct density matrices of the teleported two-qubit graph state

        Args:
            result (IBMJob.result()): result completed
            result_qrem (IBMJob.result()): QREM result completed
            post_processing (bool, optional): whether post-process intermediate qubits. Defaults to True.
            apply_mit (str, optional): mitigation mode. Defaults to 'QREM'.
            teleportation (str, optional): teleportation mode. Defaults to 'teleportation'.
            mitigate_qubits (list, optional): qubtis to mitigate in reduced_QREM. Defaults to [1,3,4,5].

        Returns:
            list: list of dictionaries of density matrices
        """
        qst_counts, pvecs = self.teleported_counts_from_result(result, result_qrem, post_processing,
                                                               teleportation = teleportation)
        if apply_mit is None:
            apply_mit = self.qrem
        #cateogrise variants of teleported state
        if self.post_processing is True:
            #Apply QREM
            if apply_mit == 'QREM':
                qst_counts, pvecs = self.apply_qrem_teleported_counts(qst_counts, pvecs,
                                                                      teleportation = teleportation, qrem = 'QREM')
                print('QREM done')
            if apply_mit == 'reduced_QREM':
                qst_counts, pvecs = self.apply_qrem_teleported_counts(qst_counts, pvecs,
                                                                      teleportation = teleportation, 
                                                                      qrem = 'reduced_QREM',
                                                                      mitigate_qubits = mitigate_qubits)
                print('reduced_QREM done')
            #calculate density matrices
            if teleportation == 'teleportation':
                #categorise
                pvecs_binned = self.bin_teleported_pvecs(qst_counts, pvecs)
                print('Categorisation done')
                rho_dict_list = []
                for i in range(self.reps):
                    rho_dict = {}
                    for gap, pairs_dict in pvecs_binned[i].items():
                        rho_dict[gap] = {pair: {} for pair in pairs_dict.keys()}
                        for pair, BS_dict in pairs_dict.items():
                            for BellState, basis_dict in BS_dict.items():
                                rho_dict[gap][pair][BellState] = Teleportation.calc_rho(basis_dict)
                        print(f'{gap} done')
                    rho_dict_list.append(rho_dict)
            
            if teleportation == 'swap':
                rho_dict_list = []
                for i in range(self.reps):
                    rho_dict = {}
                    for gap, pairs_dict in pvecs[i].items():
                        rho_dict[gap] = {}
                        for pair, basis_dict in pairs_dict.items():
                                rho_dict[gap][pair] = Teleportation.calc_rho(basis_dict)
                    rho_dict_list.append(rho_dict)
        #dynamic circuits
        else:
            if apply_mit is True:
                qst_counts, pvecs = self.apply_qrem_teleported_counts_from_dynamic_circuits(qst_counts, pvecs)
            #group common bit-strings of the ending qubits and ignore intermediate measurements
            pvecs_binned = self.bin_teleported_pvecs_from_dynamic_circuits(qst_counts, pvecs)
            rho_dict_list = []
            for i in range(self.reps):
                rho_dict = {}
                for gap, pairs_dict in pvecs_binned[i].items():
                    rho_dict[gap] = {}
                    for pair, basis_dict in pairs_dict.items():
                            rho_dict[gap][pair] = Teleportation.calc_rho(basis_dict)
                rho_dict_list.append(rho_dict)
        
        return rho_dict_list
    
    def recon_teleported_density_mats_from_multi(self, result_dict, qrem_dict=None, apply_mit=None):
        """Reconstruct density matrices of the teleported two-qubit graph state, result is in dictionary

        Args:
            result_dict (dict): dictionary of job_ids and circuit names
            qrem_dict (dict, optional): dictionary of QREM job_ids and circuit names. Defaults to None.
            apply_mit (bool, optional): mitigation mode. Defaults to None.

        Returns:
            list: list of density matrices dictionaries
        """
        qst_counts, pvecs = self.teleported_counts_from_result_list(result_dict=result_dict, qrem_dict=qrem_dict)
        if apply_mit is None:
            apply_mit = self.qrem
        if apply_mit is True:
            qst_counts, pvecs = self.apply_qrem_teleported_counts_from_dynamic_circuits(qst_counts, pvecs)

        pvecs_binned = self.bin_teleported_pvecs_from_dynamic_circuits(qst_counts, pvecs)
        rho_dict_list = []
        for i in range(self.reps):
            rho_dict = {}
            for gap, pairs_dict in pvecs_binned[i].items():
                rho_dict[gap] = {}
                for pair, basis_dict in pairs_dict.items():
                    rho_dict[gap][pair] = Teleportation.calc_rho(basis_dict)
            rho_dict_list.append(rho_dict)
        
        return rho_dict_list
    
    def bit_str_to_BellState(bit_str):
        """Map the bit_str (X-measure) as a binary string (in reversed order) into the corresponding BellState that X-measure map to

        Args:
            bit_str (str): bit-string of the X-measurements result

        Returns:
            str: one of the 4 variants of bellstates
        """
        #Base BellState
        #state_if_before_processing = BS_list_even[0].copy()
        state_column = BS_list_even[0].copy()

        bit_str_new = bit_str[::-1]
        for i in range(len(bit_str_new)):
            #state_if_before_processing.h(1)
            state_column = H1@state_column
            if bit_str_new[i] == '1':
                #state_if_before_processing.x(1)
                state_column = X1@state_column
        #state_column = Statevector(state_if_before_processing)
        #state_column = Statevector(state_if_before_processing).data
        BellState_index = 0
        for i in range(4):
            if len(bit_str) % 2 == 0:
                #if state_column.equiv(States_even[i]):
                #if np.allclose(state_column, States_even[i].data) or np.allclose(-state_column, States_even[i].data):
                if np.allclose(state_column, BS_list_even[i]) or np.allclose(-state_column, BS_list_even[i]):    
                    BellState_index = i+1
                    return f'BS_{BellState_index}'
            else:
                #if state_column.equiv(States_odd[i]):
                #if np.allclose(state_column, States_odd[i].data) or np.allclose(-state_column, States_odd[i].data):
                if np.allclose(state_column, BS_list_odd[i]) or np.allclose(-state_column, BS_list_odd[i]):    
                    BellState_index = i+1
                    return f'BS_{BellState_index}'
        
    def gen_qst_circuits(self):
        """(Deprecated) Generates (parallelised) quantum state tomography circuits

        Returns:
            dictionary: dictionary of circuits, in the batch/basis/circuit structure 
        """
        # Generate batches of groups (target edges + adjacent qubits) to perform
        # QST in parallel
        if self.batches is None:
            self.batches = self.gen_batches()

        circuits = {}  # Dictionary of groups of circuits where batches are keys
        name_list = []  # List of circuit names

        #graphstate = self.circuit.copy()
        graphstate = self.delay_circuit.copy() #commented out when using delay circuits
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
    
    def run_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None):
        """
        (Deprecated) Execute the quantum state tomography circuits

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
            
    def counts_from_result(self, result):
        """
        (Deprecated) Get counts from qiskit result as dictionary or lists of dictionaries

        """

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

        if self.reps == 1:
            return qst_counts  # Single experiment

        return qst_counts_multi  # Multiple experiments

    def group_counts(self, qst_counts):
        """
        (Deprecated) Regroups qst_counts according to tomography groups (target qubit pair
        + adjacent qubits) and convert into equivalent probability vector

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
        """
        (Deprecated) Further classify the group probability vectors according to the different
        measurement combinations on adjacent qubits

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
        """
        (Deprecated) Reconstruct the density matrices for every qubit pair for every
        measurement combination and save it as a dictionary or list of
        dictionaries

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
        circ0 = QuantumCircuit(self.device_size, name='qrem0')
        circ0.measure_all()

        circ1 = QuantumCircuit(self.device_size, name='qrem1')
        circ1.x(range(self.device_size))
        circ1.measure_all()

        self.qrem_circuits = [circ0, circ1]

        return [circ0, circ1]

    def apply_qrem(self, g_counts, g_vecs):
        """(Deprecated) Apply quantum readout error mitigation on grouped counts/probability
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
            # s for basis 'IX' and 'XI'
            s_dict['I' + basis[1]] += (pvec[0] - pvec[1] + pvec[2] - pvec[3])/3 #+ or - is only decided by whether 2nd qubit is measured
                                                                                #to be |0> or |1> because only identity is applied to 1st
                                                                                #qubit so its result is not important
            s_dict[basis[0] + 'I'] += (pvec[0] + pvec[1] - pvec[2] - pvec[3])/3
            
        # Weighted sum of basis matrices
        for basis, s in s_dict.items():
            rho += 0.25*s*pauli_n(basis)

        # Convert raw density matrix into closest physical density matrix using
        # Smolin's algorithm (2011)
        rho = Teleportation.find_closest_physical(rho)

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

def calc_teleported_negativities(rho_dict_list, post_processing = True, mode='all', teleportation = 'teleportation'):
    """Calculate negativities for each density matrix in rho_dict_list for teleported qst only

    Args:
        rho_dict_list (list): list of density matrices dictionaries
        post_processing (bool, optional): whether circuits was post-processed. Defaults to True.
        mode (str, optional): mode of output. Defaults to 'all'.
        teleportation (str, optional): teleportation mode. Defaults to 'teleportation'.

    Returns:
        list or dict: negativities of the density matrices
    """
    n_all_list = []  # Negativities for each bin per experiment
    n_mean_list = []  # Mean negativity between bins per experiment
    n_max_list = []  # Max negativity between bins per experiment
    n_min_list = []
    
    reps = len(rho_dict_list)
    for i in range(reps):
        n_all = copy.deepcopy(rho_dict_list[i])
        n_mean = {gap: {pair: 0 for pair in pairs_dict.keys()} 
                  for gap, pairs_dict in rho_dict_list[i].items()}
        n_max = copy.deepcopy(n_mean)
        n_min = copy.deepcopy(n_mean)
        
        for gap, pairs_dict in rho_dict_list[i].items():
            #post-processed
            if post_processing is True:
                if teleportation == 'teleportation':
                    for pair, bellstate_dict in pairs_dict.items():
                        n_sum = 0
                        n_list = []
                        for bellstate, rho in bellstate_dict.items():
                            n = calc_n(rho)
                            n_all[gap][pair][bellstate] = n #n_all stores each BS outcome
                            n_list.append(n)
                            n_sum += n
                        n_mean[gap][pair] = n_sum/4
                        n_max[gap][pair] = max(n_list)
                        n_min[gap][pair] = min(n_list)
                
                if teleportation == 'swap':
                    for pair, rho in pairs_dict.items():
                        n = calc_n(rho)
                        n_mean[gap][pair] = n
            #dynamic circuits
            else:
                for pair, rho in pairs_dict.items():
                    n = calc_n(rho)
                    n_mean[gap][pair] = n
        
        n_all_list.append(n_all)
        n_mean_list.append(n_mean)
        n_max_list.append(n_max)
        n_min_list.append(n_min)
    
    if post_processing is False:
        return n_mean_list
    if mode == 'all':
        return n_all_list
    elif mode == 'mean':
        return n_mean_list
    elif mode == 'max':
        return n_max_list
    elif mode == 'min':
        return n_min_list
    
    return None

def calc_negativities(rho_dict, mode='all'):
    """Obtain negativities corresponding to every density matrix in rho_dict.
    Option to obtain max, mean or all negativities between measurement
    combinations (bins)

    Args:
        rho_dict (dict): density matrices dictionary
        mode (str, optional): mode of output. Defaults to 'all'.

    Returns:
        _type_: _description_
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
            #for mode = 'all', find the negativity of each BS variant
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
    """
    (Deprecated)
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
                reduced_rho = partial_trace(rho_qiskit, [0]).data
                reduced_rho_physical = GraphState.find_closest_physical(reduced_rho)
                eigvals = np.linalg.eigvals(reduced_rho_physical)
                
                remove_idx = []
                for i in range(len(eigvals)):
                    if eigvals[i] == 0:
                        remove_idx.append(i)
                new_eigvals = np.delete(eigvals, remove_idx)
                
                S = -np.sum(new_eigvals*np.log2(new_eigvals))
                S_total += S
                
            S_mean = S/len(bns)
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
    rho_pt = ptrans(rho)
    w, _ = la.eig(rho_pt)
    n = np.sum(w[w < 0])

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

def plot_teleported_negativities_multi(backend, n_list, nmit_list=None, figsize=(6.4, 4.8)):
    """Plot average negativity across multiple experiments with error bars as std

    Args:
        backend (IBMProvider().backend): IBM backend
        n_list (list): list of negativities in dict
        nmit_list (list, optional): list of mitigated negativities in dict. Defaults to None.
        figsize (tuple, optional): size of output figure. Defaults to (6.4, 4.8).

    Returns:
        fig: matplotlib figure of the plot of negativity vs pairs
    """

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract the mean negativity and its standard deviation
    pairs = []
    for gap, pairs_dict in n_list[0].items():
        for pair in pairs_dict.keys():
            pairs.append(f'{gap}-{pair}')

    n_mean, n_std = calc_teleported_n_mean(n_list)
    
    n_mean_list = []
    n_std_list = []
    for gap, pairs_dict in n_mean.items():
        for value in pairs_dict.values():
            n_mean_list.append(value)
            
    for gap, pairs_dict in n_std.items():
        for value in pairs_dict.values():
            n_std_list.append(value)
        
    # Convert into array for plotting
    X = np.array(pairs)
    Y0 = np.array(n_mean_list)
    Y0err = np.array(n_std_list)

    # If mitigated results are included
    try:
        nmit_mean, nmit_std = calc_teleported_n_mean(nmit_list)
        n_mit_mean_list = []
        n_mit_std_list = []
        for gap, pairs_dict in nmit_mean.items():
            for value in pairs_dict.values():
                n_mit_mean_list.append(value)
            
        for gap, pairs_dict in nmit_std.items():
            for value in pairs_dict.values():
                n_mit_std_list.append(value)

        Y1 = np.array(n_mit_mean_list)
        Y1err = np.array(n_mit_std_list)
        # Order in increasing minimum negativity (QREM)
        #Y1min = Y1 - Y1err
        #idx = Y1min.argsort()#find the indicies that sort Y1min from smallest to largest
        #Y1 = Y1[idx]#then put all negativities in such order (indicies)
        #Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        idx = Y0min.argsort()

    #X = X[idx]
    #Y0 = Y0[idx]
    #Y0err = Y0err[idx]

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

def plot_teleported_negativities_multi_gap(backend, n_list, nmit_list=None, bellstate = 'BS_1', figsize=(6.4, 4.8)):
    """Plot average negativity for each gap with error bars as std

    Args:
        backend (IBMProvider.backend): backend quantum computer
        n_list (list): negativity list
        nmit_list (list, optional): mitigated negativity list. Defaults to None.
        bellstate (str, optional): which bellstate (or variant) to plot. Defaults to 'BS_1'.
        figsize (tuple, optional): size of output figure. Defaults to (6.4, 4.8).

    Returns:
        fig: output figure, plot negativity vs gaps for specific bellstate
    """
    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract the mean negativity and its standard deviation
    gaps = []
    for gap in n_list[0].keys():
        gaps.append(f'{gap}')
    #find the mean and standard error the negativities for this specific bell state at different gaps
    n_mean, n_std = calc_teleported_n_mean_gap(n_list, bellstate)
    
    n_mean_list = []
    n_std_list = []
    # Convert to list
    for gap, mean in n_mean.items():
        n_mean_list.append(mean)
            
    for gap, std in n_std.items():
        n_std_list.append(std)
        
    # Convert into array for plotting
    X = np.array(gaps)
    Y0 = np.array(n_mean_list)
    Y0err = np.array(n_std_list)

    # If mitigated results are included
    try:
        nmit_mean, nmit_std = calc_teleported_n_mean_gap(nmit_list, bellstate)
        n_mit_mean_list = []
        n_mit_std_list = []
        for gap, mean in nmit_mean.items():
            n_mit_mean_list.append(mean)
            
        for gap, std in nmit_std.items():
            n_mit_std_list.append(std)

        Y1 = np.array(n_mit_mean_list)
        Y1err = np.array(n_mit_std_list)
        # Order in increasing minimum negativity (QREM)
        #Y1min = Y1 - Y1err
        #idx = Y1min.argsort()#find the indicies that sort Y1min from smallest to largest
        #Y1 = Y1[idx]#then put all negativities in such order (indicies)
        #Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        idx = Y0min.argsort()

    #X = X[idx]
    #Y0 = Y0[idx]
    #Y0err = Y0err[idx]

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

    ax.set_xlabel("Gap")
    ax.set_ylabel("Negativity")
    #ax.set_title(f"Native-graph state negativities ({backend.name()})")
    ax.set_title(backend.name)

    return fig

def teleported_negativities_multi_gap_data(n_list, nmit_list=None, bellstate = 'BS_1'):
    """Plot average negativity for each gap with error bars as std

    Args:
        n_list (list): negativity list
        nmit_list (list, optional): mitigated negativity list. Defaults to None.
        bellstate (str, optional): which bellstate (or variant) to plot. Defaults to 'BS_1'.

    Returns:
        three lists: data of negativities, mitigated negativities and standard errors
    """
    # Extract the mean negativity and its standard deviation
    gaps = []
    for gap in n_list[0].keys():
        gaps.append(f'{gap}')

    n_mean, n_std = calc_teleported_n_mean_gap(n_list, bellstate)
    
    n_mean_list = []
    n_std_list = []
    #Convert to list
    for gap, mean in n_mean.items():
        n_mean_list.append(mean)
            
    for gap, std in n_std.items():
        n_std_list.append(std)
        
    # Convert into array for plotting
    X = np.array(gaps)
    Y0 = np.array(n_mean_list)
    Y0err = np.array(n_std_list)

    # If mitigated results are included
    try:
        nmit_mean, nmit_std = calc_teleported_n_mean_gap(nmit_list, bellstate)
        n_mit_mean_list = []
        n_mit_std_list = []
        for gap, mean in nmit_mean.items():
            n_mit_mean_list.append(mean)
            
        for gap, std in nmit_std.items():
            n_mit_std_list.append(std)

        Y1 = np.array(n_mit_mean_list)
        Y1err = np.array(n_mit_std_list)
        # Order in increasing minimum negativity (QREM)
        #Y1min = Y1 - Y1err
        #idx = Y1min.argsort()#find the indicies that sort Y1min from smallest to largest
        #Y1 = Y1[idx]#then put all negativities in such order (indicies)
        #Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        idx = Y0min.argsort()

    #X = X[idx]
    #Y0 = Y0[idx]
    #Y0err = Y0err[idx]
    try:
        return X, Y0, Y0err, Y1, Y1err
    except:
        return X, Y0, Y0err

def plot_negativities_multi(backend, n_list, nmit_list=None, figsize=(6.4, 4.8)):
    """
    (Deprecated) Plot average negativity across multiple experiments with error bars as std

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
    """
    (Deprecated) Plot negativity vs. CNOT error
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

def calc_teleported_n_mean(n_list):
    """find mean and standard error of the negativites over trials/reps

    Args:
        n_list (list): list of negativities

    Returns:
        two dictionaries: mean negativity and standard error of negativity dictionaries
    """
    reps = len(n_list)
    #for each qubit pair store trials on BS1 negativity 
    n_dict = {gap: {pair: [n_list[i][gap][pair]['BS_1'] for i in range(reps)] 
                    for pair in pairs_dict.keys()} 
              for gap, pairs_dict in n_list[0].items()}
    # mean negativity on each pair
    n_mean = {gap: {pair: np.mean(negativities)
                    for pair, negativities in pairs_dict.items()}
              for gap, pairs_dict in n_dict.items()}
    # standard error on each pair
    n_std_err = {gap: {pair: np.std(negativities)/np.sqrt(reps) 
                       for pair, negativities in pairs_dict.items()} 
              for gap, pairs_dict in n_dict.items()}

    return n_mean, n_std_err

def calc_teleported_n_mean_gap(n_list, bellstate):
    """Calculate mean negativity dict for gap only

    Args:
        n_list (list): list of negativities dictionary
        bellstate (str): name of which bellstate

    Returns:
        two dictioanries: mean and standard errors of the negativity on bellstate
    """
    reps = len(n_list)
    # construct the dictionary
    n_dict = {gap: [] for gap in n_list[0].keys()}
    #if dynamic circuit
    if bellstate is None:
        for i in range(reps):
            for gap, pairs_dict in n_list[i].items():
                for pair, value in pairs_dict.items():
                    n_dict[gap].append(value)
    #take average of 4 variants
    elif bellstate == 'ignore':
        for i in range(reps):
            for gap, pairs_dict in n_list[i].items():
                for pair, bellstates_dict in pairs_dict.items():
                    if gap == 1:
                        n_dict[gap].append((bellstates_dict['BS_1']+bellstates_dict['BS_2'])/2)
                    elif gap%2 == 0:
                        n_dict[gap].append((bellstates_dict['BS_1']+bellstates_dict['BS_2']+
                                            bellstates_dict['BS_3']+bellstates_dict['BS_4'])/4)
    #find negativity of each variant bellstate
    else:
        for i in range(reps):
            for gap, pairs_dict in n_list[i].items():
                for pair, bellstates_dict in pairs_dict.items():
                    n_dict[gap].append(bellstates_dict[bellstate])
    
    n_mean = {gap: np.mean(negativities) for gap, negativities in n_dict.items()}
    
    n_std_err = {gap: np.std(negativities)/np.sqrt(len(negativities)) for gap, negativities in n_dict.items()}

    return n_mean, n_std_err

def calc_n_mean(n_list):
    """
    (Deprecated) Calculate mean negativity dict from lists of negativity dicts
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
    """
    (Deprecated) Plot the number of QST patches for each available device
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
