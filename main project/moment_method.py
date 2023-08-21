# Standard libraries
from math import factorial
from datetime import datetime
import queue as Q
import itertools
import copy

#other installed libraries
from numpy import log2,array,matrix,kron
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
import networkx as nx

# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute, transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import InstructionDurations
from qiskit_ibm_provider import IBMProvider
#from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix, Operator, PauliList

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

#Pauli matrices
X=matrix([[0,1],[1,0]])
Y=matrix([[0,-1j],[1j,0]])
Z=matrix([[1,0],[0,-1]])
I=matrix([[1,0],[0,1]])

#Define conversion from Paulis to matrices or strings
P2M_LOOKUP={(0,0):I,(0,1):Z,(1,0):X,(1,1):Y,}
P2S_LOOKUP={(0,0):'I',(0,1):'Z',(1,0):'X',(1,1):'Y',}

# ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]

def hamming_weight(n):
    """return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)

    Args:
        n (int): any integer

    Returns:
        int: number of ones of the integer n after converted to binary
    """
    c = 0
    while n:
        c += POPCOUNT_TABLE16[n & 0xffff]
        n >>= 16
    return c

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
    
def sum_error(a, b):
    return a + b - a*b

class MomentMethod(Free_EntangleBase):
    """ Class to run and analyse Energy estimation of graph state w.r.t Hamiltonians

    Args:
        Free_EntangleBase (Class): parent class object
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
        self.Hamiltonian = self.gen_Hamiltonian()
        self.Hamiltonian_powers = None
        self.TPB_sets = None
        self.order = None

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
            connected_edges = []
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
    
    def gen_ghz_circuit(self, nodes, source=None, output_error=False):
        """(Deprecated)"""
        self.nqubits = nodes

        #circ = QuantumCircuit(nodes)
        circ = QuantumCircuit(self.device_size)

        # If source is None
        if source is None:
            source, cx_instr, initial_layout, depth, terror_dict = self.find_opt_source(
                nodes)
        # If source qubit is specified
        else:
            cx_instr, initial_layout, depth, terror_dict = self.gen_circ_instr(
                nodes, source)
        self.ghz_edges = cx_instr
    
        # Construct circuit in Qiskit
        circ.h(initial_layout[0])
        for edge in cx_instr:
            circ.cx(*edge)

        self.ghz_circuit = circ
        self.ghz_size = nodes
        self.initial_layout = initial_layout
        self.qubits_to_connect = sorted(initial_layout)

        if output_error is True:
            return circ, initial_layout, terror_dict

        return circ, initial_layout
    
    def find_opt_source(self, nodes, error_key='cumcx'):
        """(Deprecated)"""
        cx_instr = None
        initial_layout = None
        min_depth = self.device_size
        min_terror = self.device_size*1000000

        for q in range(self.device_size):
            instr, layout, depth, terror_dict = self.gen_circ_instr(nodes, q)
            # If CNOT depth is less than or equal to
            if depth <= min_depth:
                # If total error is less than
                #if terror_dict[error_key] < min_terror or depth < min_depth:
                source = q
                cx_instr = instr
                initial_layout = layout
                min_depth = depth
                min_terror = terror_dict[error_key]

        return source, cx_instr, initial_layout, min_depth, terror_dict

    def gen_circ_instr(self, nodes, source, mapped=False):
        """(Deprecated) Modified Dijkstra's algorithm"""

        terror_dict = {'cumcx': 0,
                       'meancx': 0,
                       't1': 0,
                       't2': 0}

        length = {q: self.device_size for q in range(self.device_size)}
        next_depth = {}
        path = {q: [] for q in range(self.device_size)}
        #degree_visited = {q: 0 for q in range(self.nqubits)}
        degree = dict(self.graph.degree)
        degree_unvisited = degree.copy()
        error = {q: 1 for q in range(self.device_size)}

        length[source] = 0
        path[source] = [source]
        error[source] = 0

        visited = []
        unvisited = length.copy()

        for i in range(nodes):
            # Find minimum length (CNOT depth) unconnected nodes
            lmin = min(unvisited.values())
            #unvisited_lmin = {key: value for key, value in unvisited.items() if value == lmin}
            u_lmin = [key for key, value in unvisited.items() if value == lmin]
            # Pick potential nodes with largest degree
            degree_lmin = {key: degree_unvisited[key] for key in u_lmin}
            dmax = max(degree_lmin.values())
            u_lmin_dmax = [key for key,
                           value in degree_lmin.items() if value == dmax]
            # Pick node with lowest propagated CNOT error
            u = min(u_lmin_dmax, key=error.get)
            degree_unvisited[u] -= 1
            # Update error dict
            terror_dict['cumcx'] += error[u]
            terror_dict['t1'] += self.backend.properties().t1(u)
            terror_dict['t2'] += self.backend.properties().t2(u)

            visited.append(u)
            del unvisited[u]
            next_depth[u] = length[u] + 1
            

            for v in self.connections[u]:
                degree_unvisited[v] -= 1
                alt = length[u] + 1
                if alt < length[v]:
                    unvisited[v] = length[v] = alt
                    path[v] = path[u] + [v]
                    error[v] = sum_error(
                        error[u], self.edge_params[tuple(sorted((u, v)))])
                    
            try:
                u_prev = path[u][-2]
                next_depth[u_prev] = length[u] + 1
                #print(next_depth)
                
                for v_prev in self.connections[u_prev]:
                    if v_prev in unvisited:
                        connected = set(self.connections[v_prev]) - set(unvisited)
                        length[v_prev] = unvisited[v_prev] = next_depth[min(connected, key=next_depth.get)]
            except: pass

        cx_instr = []
        
        if mapped is True:
            qubits = []
            edges = []
            depths = []
            for q in visited[1:]:
                qubits.append(q)
                edges.append(path[q][-2:])
                depths.append(length[q])
                
            return qubits, edges, depths
                
        
        for q in visited[1:]:
            c, t = path[q][-2:]
            #cx_instr.append((visited.index(c), visited.index(t)))
            cx_instr.append((c, t))
        initial_layout = visited
        depth = length[u]   

        return cx_instr, initial_layout, depth, terror_dict
    
    
    def gen_Hamiltonian(self):
        """Generate hamiltonian info dictionary where keys are X&Z occupation integers and value are coefficient of this pauli list.
        Should expect to return {(1010,0100):[-1/4], (0101,0010):[-1/4],...} etc

        Returns:
            dict: Hamiltonian information
        """
        Hamiltonian_basis = []
        #find the stablizers (hence elements of Hamiltonian of GraphState) in pauli strings, order from qubit 0 to qubit n
        for qubit, neighbours in self.connections.items():
            pauli_string = 'I'*self.device_size
            pauli_string = pauli_string[:qubit] + 'X' + pauli_string[qubit+1:]
            for neighbour in neighbours:
                pauli_string = pauli_string[:neighbour] + 'Z' + pauli_string[neighbour+1:]
            
            true_pauli_string = ''
            for qubit in self.qubits_to_connect:
                true_pauli_string += pauli_string[qubit]
            Hamiltonian_basis.append(true_pauli_string)
        
        Hamiltonian_info = {}
        for pauli_string in Hamiltonian_basis:
            X_integer = 0  #left/right in tuple is the integer represent X/Z-paulis position after converting to binary
                           #Note if both X,Z == 1 at same position then it represents a Y
            Z_integer = 0
            for i in range(len(pauli_string)):
                # binary to integer conversion (Largest indice on left and Smallest on right, i.e. 100 = 4)
                if pauli_string[i] == 'X':
                    X_integer += 2**(self.nqubits -1 - i)
                elif pauli_string[i] == 'Z':
                    Z_integer += 2**(self.nqubits -1 - i)
            Hamiltonian_info[(X_integer, Z_integer)] = -np.ones(1)/self.nqubits #Initial coefficient always -1
            #number of pauli strings in Hamiltonian of graphstate = number of qubits. Therefore normalisation / self.nqubits
        
        return Hamiltonian_info
    
    def gen_GHZ_Hamiltonian(self):
        """
        (Deprecated) Generate GHZ hamiltonian info dictionary where keys are X&Z occupation integers and value are coefficient of this pauli list.
        Should expect to return {(1010,0100):[-1/4], (0101,0010):[-1/4],...} etc
        """
        
        #find the stablizers (hence elements of Hamiltonian of GHZ state) in pauli strings, order from qubit 0 to qubit n
        Hamiltonian_basis = []
        for edge in self.ghz_edges:
            pauli_string = 'I'*self.device_size
            pauli_string = pauli_string[:edge[0]] + 'Z' + pauli_string[edge[0]+1:]
            pauli_string = pauli_string[:edge[1]] + 'Z' + pauli_string[edge[1]+1:]
            true_pauli_string = ''
            for qubit in self.qubits_to_connect:
                true_pauli_string += pauli_string[qubit]
            Hamiltonian_basis.append(true_pauli_string)
        Hamiltonian_basis.append('X'*self.nqubits)
        
        Hamiltonian_info = {}
        for pauli_string in Hamiltonian_basis:
            X_integer = 0  #left/right in tuple is the integer represent X/Z-paulis position after converting to binary
                           #Note if both X,Z == 1 at same position then it represents a Y
            Z_integer = 0
            for i in range(len(pauli_string)):
                # binary to integer conversion (Largest indice on left and Smallest on right, i.e. 100 = 4)
                if pauli_string[i] == 'X':
                    X_integer += 2**(self.nqubits -1 - i)
                elif pauli_string[i] == 'Z':
                    Z_integer += 2**(self.nqubits -1 - i)
            Hamiltonian_info[(X_integer, Z_integer)] = -np.ones(1)/len(Hamiltonian_basis) #Initial coefficient always -1
            #number of pauli strings in Hamiltonian of graphstate = number of qubits. Therefore normalisation / self.nqubits
        
        return Hamiltonian_info
    
    def to_string(self, Hamiltonian, index = 0):
        """Convert a Hamiltonian dictionary into pauli-string expressions

        Args:
            Hamiltonian (dict): Hamiltonian dict information H^n
            index (int, optional): order of H^n. Defaults to 0.

        Returns:
            _type_: _description_
        """
        sz = self.nqubits
        S = ''
        for (x,z),w in Hamiltonian.items():
            s = f'{w[index]}*' + ''.join(P2S_LOOKUP[(bool(x&2**ix),bool(z&2**ix))] for ix in range(sz))[::-1]
            if (not s[0]=='-') and len(S):
                s = '+' + s
            S += s
        if not len(S):
            return '0'
        
        return S
    
    def Hamiltonian_mul(self, H1, H2):
        """Multiply 2 Hamiltonians at (in principle) different powers, should return new dict

        Args:
            H1 (dict): Hamiltonian info dict 1
            H2 (dict): Hamiltonian info dict 2

        Returns:
            dict: new hamiltonian =H1*H2
        """
        new={}
        #iterate over all pairs of terms
        for (x1,z1),w1 in H1.items():
            for (x2,z2),w2 in H2.items():
                #find locations where one operator has an X (or Y) gate and the other does not, these will be the locations of X (or Y) gates in the new operator
                x3=x1^x2 #(^ represent binary bitwise addition %2)
                #find locations where one operator has an Z (or Y) gate and the other does not, these will be the locations of Z (or Y) gates in the new operator
                z3=z1^z2
                #adjust the coefficient of the new string, each call to hamming_weight counts the number of locations at which the given pair of matrices are found
                s=(1j)**( hamming_weight(x1&(~z1)&(~x2)&z2)#XZ
                         +hamming_weight(x1&z1&x2&(~z2))#YX
                         +hamming_weight((~x1)&z1&x2&z2)#ZY
                         -hamming_weight((~x1)&z1&x2&(~z2))#ZX
                         -hamming_weight(x1&(~z1)&x2&z2)#XY
                         -hamming_weight(x1&z1&(~x2)&z2))#YZ
                #assign the calculated strings and weights to the new operator (add them to the current entry if it already exists)
                new[(x3,z3)]=new.get((x3,z3),0)+w1*w2*s
        return new
    
    def combine_Hamiltonian_info(self, H1, H2):
        """Combine 2 Hamiltonian info dict in different orders, expect to return 
        dic = {(1010,0100):[-1/4, 1/16], (0101,0010):[-1/4, 0], (....,....): [0, 1/16], ...} etc

        Args:
            H1 (_type_): Hamiltonian info dict 1
            H2 (_type_): Hamiltonian info dict 2

        Returns:
            dict: new Hamiltonian info dict with keys are all existing pauli strings of H1 and H2
            and coefficients as values
        """
        n_s = len(list(H1.values())[0])
        n_o = 1
        dic = {key:array((H1[key].tolist() if key in H1 else [0]*n_s)+
                         (H2[key].tolist() if key in H2 else [0]*n_o))
               for key in set(H1).union(H2)}
        return dic
        
    def gen_Hamiltonian_powers(self, order = 4):
        """Generate the big Hamiltonian dict for all orders H^1 to H^n, each element in the array is a coeff for that order

        Args:
            order (int, optional): highest order of H^n. Defaults to 4.

        Returns:
            dict: all pauli-strings and coefficients from H^1 to H^n
        """
        self.order = order
        Hamiltonian_powers = self.Hamiltonian.copy() #copy initial (1st order) Hamiltonian info dict
        Hprev = self.Hamiltonian.copy()
        
        for i in range(1, order):
            # Calculate H^(i+1) and combine with H^i to form larger arrays, until iterate to largest order
            Hnext = self.Hamiltonian_mul(Hprev, self.Hamiltonian)
            Hamiltonian_powers = self.combine_Hamiltonian_info(Hamiltonian_powers, Hnext)
            Hprev = Hnext.copy()
        
        self.Hamiltonian_powers = Hamiltonian_powers
        return Hamiltonian_powers
    
    
    def gen_TPB_set(self):
        """find TPB sets where within each TPB set each pauli string (integer) Qubit-wise commutes, 
        First element in each TPB set is always the one to be measured
        expect to return TPB_sets = [[(23,10), (52,38),(18,39)], [......],...]

        Returns:
            list: list of lists of grouped TPB sets
        """
        # take all pauli-strings (in X-Z integer form) into list
        operator_integers = []
        for key in self.Hamiltonian_powers:
            operator_integers.append(key)
        #initialize to first operator
        TPB_sets = [[operator_integers[0], operator_integers[0]]]
        for operator in operator_integers[1:]:
            for i in range(len(TPB_sets)):
                BWC = True #default to true
                #set to False if not BWC
                if self.check_BWC(operator, TPB_sets[i][0]) is False:
                    BWC = False
                #add to this TPB set if BWC to all of the rest in this group and check next operator
                if BWC is True:
                    TPB_sets[i].append(operator)
                    TPB_sets[i][0] = (TPB_sets[i][0][0]|operator[0], TPB_sets[i][0][1]|operator[1])
                    break
            #if cannot add to any, create its own
            if BWC is False:
                TPB_sets.append([operator, operator])
        self.TPB_sets = TPB_sets
        
        #TPB_sets = [[operator_integers[0]]] #initial case, put first pauli string
        #for operator in operator_integers[1:]:
        #    for i in range(len(TPB_sets)):
        #        BWC = True
        #        for basis in TPB_sets[i]:
                    #check if all basis in this TPB set QWC with the operator, 
                    #if it is, BWC remains true and it will be added, if not go to next set
        #            if self.check_BWC(operator, basis) is False:
        #                BWC = False
        #                break
        #        if BWC is True:
                    # if true then add this to current TPB set, then move to next operator
        #            TPB_sets[i].append(operator)
        #            break
        #    if BWC is False:
                # if BWC is always found false i.e. none of existing TPB sets accepts it then create a new set for it to live
        #        TPB_sets.append([operator])
        
        #for i in range(len(TPB_sets)):
        #    to_be_measured = TPB_sets[i][0]
        #    for operator in TPB_sets[i]:
                #Find the measurement basis such that it has least Identities and QWC all in this TPB set
        #        x1 = to_be_measured[0]
        #        z1 = to_be_measured[1]
        #        x2 = operator[0]
        #        z2 = operator[1]
        #        to_be_measured = (x1|x2, z1|z2)
        #    TPB_sets[i] = [to_be_measured] + TPB_sets[i]
        
        #self.TPB_sets = TPB_sets
        
        measurement_mapping = {}
        # Create a dict that maps the hamiltonian basis strings to the basis actually being measured
        for group in TPB_sets:
            measurement_basis = group[0]
            for hamiltonian_basis in group[1:]:
                measurement_mapping[hamiltonian_basis] = measurement_basis
        self.measurement_mapping = measurement_mapping
        
        return TPB_sets
    
    def gen_delay_circuit(self, t, increment, dynamic_decoupling=False):
        '''
        (Deprecated) Generate delay circuit based on Graphstate circuit
        '''
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
        """(Deprecated)"""
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
        '''(Deprecated) Generate double X pulse dynamical decoupling circuit'''
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
        '''(Deprecated) Generate hahn echo dynamical decoupling circuit'''
        tdelay = t - self.tx
        padding = self.format_delays(0.5*tdelay, unit = 'dt')
        
        self.delay_circuit = self.circuit.copy()
        
        self.delay_circuit.barrier()
        self.delay_circuit.delay(padding)
        self.delay_circuit.x(range(self.nqubits))
        self.delay_circuit.delay(padding)
        
    def format_delays(self, delays, unit='ns'):
        """(Deprecated)"""
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
    
    def gen_measurement_circuits(self, delay = False):
        """For each measurement basis, generate the measurement circuit

        Args:
            delay (bool, optional): whether add delays to circuit. Defaults to False.

        Returns:
            dict: dictionary of circuits
        """
        measure_integers_list = [group[0] for group in self.TPB_sets.copy()]
        #Convert integer to pauli string
        #measure_pauli_basis_list = [self.operator_to_string(integer) for integer in measure_integers_list]
        
        if delay is True:
            graphstate = self.delay_circuit.copy()
        else:
            graphstate = self.circuit.copy() #commented out when using delay circuits
        #graphstate.barrier()
        
        moment_circuits = {}
        name_list = []
        
        #for each measurement operator change to corresponding basis
        for operator in measure_integers_list:
            basis = self.operator_to_string(operator)
            circ = graphstate.copy(basis)
            for i in range(len(basis)):
                if basis[i] == 'X':
                    circ.h(self.qubits_to_connect[i])
                elif basis[i] == 'Y':
                    circ.sdg(self.qubits_to_connect[i])
                    circ.h(self.qubits_to_connect[i])
            cr = ClassicalRegister(self.nqubits)
            circ.add_register(cr)
            circ.measure(self.qubits_to_connect, cr)
            
            moment_circuits[basis] = circ
            #name_list.append(circ.name)
            name_list.append(operator)
            
        self.moment_circuits = moment_circuits
        self.name_list = name_list #record the circuit names, where names are the basis names
        #self.initial_layout = list(range(self.device_size))
        return moment_circuits
    
    def run_moment_circuits(self, order=4, reps=1, shots=4096, qrem = False, sim=None):
        """execute the circuits with target repetitions and shots, qrem is added if needed

        Args:
            order (int, optional): highest order of the moments. Defaults to 4.
            reps (int, optional): total repetitions on the moment circuits. Defaults to 1.
            shots (int, optional): shots per each circuit. Defaults to 4096.
            qrem (bool, optional): whether run QREM circuit together. Defaults to False.
            sim (_type_, optional): simulator mode. Defaults to None.

        Returns:
            IBMJob: job submitted to IBM Backend quantum computers
        """
        #Initialize the prerequired objects to run
        if self.Hamiltonian_powers is None:
            self.gen_Hamiltonian_powers(order = order)
        if self.TPB_sets is None:
            self.gen_TPB_set()
        moment_circuits = self.gen_measurement_circuits()
        
        # Convert circuits dict into list form
        circ_list = []
        for circ in moment_circuits.values():
            circ_list.append(circ)
        # Extend circuit list by number of repetitions   
        circ_list_multi = []
        for i in range(reps):
            for circ in circ_list:
                name_ext = circ.name + f'-{i}'
                circ_list_multi.append(circ.copy(name_ext))
        circ_list = circ_list_multi
        #add QREM circuits if needed
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)
        #actual machine
        if sim is None:
            job = execute(circ_list, backend=self.backend, shots=shots)
            #circ_list_transpiled = transpile(circ_list, backend = self.backend)
            #job = self.backend.run(circ_list_transpiled, shots=shots)
        #noiseless simulator
        elif sim == "ideal":
            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend, 
                          initial_layout=list(range(self.device_size)),
                          shots=shots)
        #noisy simulator
        elif sim == "device":
            # Obtain device and noise model parameters
            #noise_model = NoiseModel.from_backend(self.backend)

            properties = self.backend.properties()
            noise_model = NoiseModel.from_backend_properties(properties)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            backend = Aer.get_backend('aer_simulator')
            job = execute(circ_list, backend=backend,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        return job
    
    def counts_from_result(self, result, order = 4):
        """Obtain all counts from result, expect to return [{'XIX':{'000':33,'010':12,...}, 'ZZY':{...}}, {...}]]
        and probability vectors. Note bit string reads reversely but pvecs are normal

        Args:
            result (IBMJob.result()): job result completed
            order (int, optional): highest order of the moments. Defaults to 4.

        Returns:
            list: list of counts dictionaries
        """
        #initialize required objects
        self.order = order
        if self.name_list is None:
            self.gen_Hamiltonian_powers(order = order)
            self.gen_TPB_set()
            self.gen_measurement_circuits()
        
        #if result is in a circuit name-job_id form
        if not isinstance(result, dict):
            if self.reps is None:
                self.reps = int(len(result.results)/len(self.name_list))
                self.shots = result.results[0].shots
        
            basis_counts_list = []
            for i in range(self.reps):
                basis_counts = {basis: {} for basis in self.name_list}
                for operator in self.name_list:
                    basis = self.operator_to_string(operator)
                    name_ext = basis + f'-{i}'
                    counts = result.get_counts(name_ext)
                    #basis_counts[basis] = counts
                    basis_counts[operator] = counts
            
                basis_counts_list.append(basis_counts)
        
        #x = 0
        #for i in range(self.reps):
        #    basis_counts = {basis: {} for basis in self.name_list}
        #    for operator in self.name_list:
        #        distribution = result.quasi_dists[x]
        #        counts_bit_str = {}
        #        for integer, prob in distribution.items():
        #            counts_bit_str[bin(integer)[2:]] = prob*self.shots
        #        basis_counts[operator] = counts_bit_str
        #        x += 1
        #    basis_counts_list.append(basis_counts)
        
        #if result is just one single object
        else:
            if self.reps is None:
                provider = IBMProvider()
                example_id = list(result.values())[-1]
                job = provider.backend.retrieve_job(example_id)
                self.reps = int(job.result().to_dict()['results'][-3]['header']['name'][-1])+1
                print(self.reps)
                self.shots = job.result().results[0].shots
                print(self.shots)
            #get all results from job ids
            results_list = []
            for v in result.values():
                job = provider.backend.retrieve_job(v)
                results_list.append(job.result())
            #store counts into dictionaries
            basis_counts_list = []
            for i in range(self.reps):
                basis_counts = {basis: {} for basis in self.name_list}
                for operator in self.name_list:
                    basis = self.operator_to_string(operator)
                    name_ext = basis + f'-{i}'
                    for res in results_list:
                        try:
                            counts = res.get_counts(name_ext)
                            break
                        except:
                            pass
                    basis_counts[operator] = counts
                    print(f'{operator} done')
                basis_counts_list.append(basis_counts)
        
        return basis_counts_list
    
    def zipped_counts_from_result(self, result, order = 4, start = 0, end = 8):
        """Obtain all counts from result, expect to return [{'XIX':{'000':33,'010':12,...}, 'ZZY':{...}}, {...}]]
        and probability vectors. Note bit string reads reversely but pvecs are normal
        (this time keys of counts are in intger, not bit-string, to save memory)

        Args:
            result (IBMJob.result() or dict): job results completed
            order (int, optional): highest order of moments. Defaults to 4.
            start (int, optional): start index+1 of repetitions to analyse. Defaults to 0.
            end (int, optional): end index+1 of repetitions to analyse. Defaults to 8.

        Returns:
            _type_: _description_
        """
        #Initialize required objects
        self.order = order
        if self.name_list is None:
            self.gen_Hamiltonian_powers(order = order)
            self.gen_TPB_set()
            self.gen_measurement_circuits()
        #if result is just one single object
        if not isinstance(result, dict):
            if self.reps is None:
                self.reps = int(len(result.results)/len(self.name_list))
                self.shots = result.results[0].shots
        
            basis_counts_list = []
            for i in range(self.reps):
                basis_counts = {basis: {} for basis in self.name_list}
                for operator in self.name_list:
                    basis = self.operator_to_string(operator)
                    name_ext = basis + f'-{i}'
                    counts = result.get_counts(name_ext)
                    counts_zipped = {}
                    for bit_string, count in counts.items():
                        idx = int(bit_string[::-1],2)
                        counts_zipped[idx] = count #counts keys are in integer to save memory
                    #basis_counts[basis] = counts_zipped
                    basis_counts[operator] = counts_zipped 
                
                basis_counts_list.append(basis_counts)
        
        #x = 0
        #for i in range(self.reps):
        #    basis_counts = {basis: {} for basis in self.name_list}
        #    for operator in self.name_list:
        #        distribution = result.quasi_dists[x]
        #        counts = {}
        #        for integer, prob in distribution.items():
        #            bit_str = bin(integer)[2:]
        #            idx = int(bit_str[::-1],2)
        #            counts[idx] = prob*self.shots
        #        basis_counts[operator] = counts
        #        x += 1
        #    basis_counts_list.append(basis_counts)
        
        #if result is in a circuit name-job_id form
        if isinstance(result, dict):
            results_list = []
            provider = IBMProvider()
            for v in result.values():
                job = provider.backend.retrieve_job(v)
                results_list.append(job.result())
                #results_list.append(job)
                print(f'{job} result get')
                
            if self.reps is None:
                example_result = results_list[-1]
                self.reps = int(example_result.results[-3].header.name[-1]) + 1
                print(self.reps)
                self.shots = example_result.results[0].shots
                print(self.shots)
                
            basis_counts_list = []
            for i in range(self.reps):
            #for i in range(start, end):
                basis_counts = {operator: {} for operator in self.name_list}
                for operator in self.name_list:
                    basis = self.operator_to_string(operator)
                    name_ext = basis + f'-{i}'
                    for res in results_list:
                        try:
                            counts = res.get_counts(name_ext)
                            #counts = res.result().get_counts(name_ext)
                            break
                        except:
                            pass
                    counts_zipped = {}
                    for bit_string, count in counts.items():
                        idx = int(bit_string[::-1],2)
                        counts_zipped[idx] = count
                    basis_counts[operator] = counts_zipped
                    print(f'{operator} done')
                print(f'rep {i} done')
                basis_counts_list.append(basis_counts)
        
        return basis_counts_list
    
    def pvecs_from_result(self, result, order = 4):
        """obtain probability vectors from results

        Args:
            result (IBMJob.result() or dict): job result completed
            order (int, optional): highest order of moments. Defaults to 4.

        Returns:
            list: list of pvecs dictionaries
        """
        #Initialize required objects
        self.order = order
        if self.name_list is None:
            self.gen_Hamiltonian_powers(order = order)
            self.gen_TPB_set()
            self.gen_measurement_circuits()
        
        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))
            self.shots = result.results[0].shots
        try:  # Try to obtain QREM results
            result.get_counts('qrem0')
            self.qrem = True
        except:
            self.qrem = False
        
        basis_pvecs_list = []
        for i in range(self.reps):
            #store as pvecs
            basis_pvecs = {basis: np.zeros(2**self.nqubits) for basis in self.name_list}
            #basis_pvecs = {basis: lil_matrix((1, 2**self.nqubits))for basis in self.name_list}
            for operator in self.name_list:
                basis = self.operator_to_string(operator)
                name_ext = basis + f'-{i}'
                counts = result.get_counts(name_ext)
                for bit_str, count in counts.items():
                    idx = int(bit_str[::-1], 2)
                    basis_pvecs[operator][idx] += count
                    #basis_pvecs[basis][(0, idx)] += count
                basis_pvecs[operator] /= self.shots

            basis_pvecs_list.append(basis_pvecs)
        
        # Find calibration matrices
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
                #M /= np.linalg.det(M)

            self.M_list = M_list
            
        return basis_pvecs_list
    
    def gen_M_list(self, result):
        """find the list of single-qubit calibration matrices

        Args:
            result (IBMJob.result()): job result completed
        """
        #get QREM result counts
        qrem_counts = [result.get_counts('qrem0'), result.get_counts('qrem1')]
        #Find calibration matrices
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
            #M /= np.linalg.det(M)

        self.M_list = M_list
            
    def apply_qrem(self, pvecs_list):
        """Apply Quantum Readout Error Mitigation to all probability vectors

        Args:
            pvecs_list (list): list of pvec dictionaries

        Returns:
            list: Error mitigated pvecs_list
        """
        pvecs_mit_list = []
        # Invert n-qubit calibration matrix
        M_inv = la.inv(self.calc_M_multi(self.qubits_to_connect)) #order in 0,1,2,3...
        for i in range(self.reps):
            pvecs_mit = pvecs_list[i].copy()
            for operator, pvec in pvecs_list[i].items():
                # "Ideal" probability vector
                pvec_mit = np.matmul(M_inv, pvec)
                pvec_mit_physical = find_closest_pvec(pvec_mit)
                pvecs_mit[operator] = pvec_mit_physical

            pvecs_mit_list.append(pvecs_mit)

        return pvecs_mit_list
    
    def apply_zipped_reduced_qrem(self, counts_list, mitigate_qubits = [1,3,4,5,6], threshold = 0.1):
        """Apply Reduced Quantum Readout Error Mitigation to zipped counts

        Args:
            counts_list (list): list of counts dictionaries
            mitigate_qubits (list, optional): list of qubits indicies to mitigate. Defaults to [1,3,4,5,6].
            threshold (float, optional): minimum value of a bit-string below it will be zero out. Defaults to 0.1.

        Returns:
            list: error mitigated counts_list
        """
        counts_list_mit = copy.deepcopy(counts_list)
        for i in range(self.reps):
            for operator, counts in counts_list[i].items():
                corrected_counts = copy.deepcopy(counts)
                #iterate over each qubit
                for q in mitigate_qubits:
                    idx = self.qubits_to_connect.index(q)
                    calibration_M = la.inv(self.M_list[q])
                    applied_names = set([])
                    corrected_int = [k for k in corrected_counts.keys()]
                    for bit_string_int in corrected_int:
                        bit_string = bin(bit_string_int)[2:].zfill(self.nqubits)
                        bit_string_int_orig = int(bit_string[::-1],2)
                        bit_string_list = list(bit_string)
                        bit_string_list[idx] = '_'
                        name = "".join(bit_string_list)
                        #check if the bit-string (except digit q) is already been corrected
                        if name not in applied_names:
                            applied_names.add(name)
                            #check the digit is 0 or 1, then flip it
                            if (bit_string_int_orig & (1 << idx)) != 0:
                                bit_string_list[idx] = '0'
                            else:
                                bit_string_list[idx] = '1'
                            bit_string_flip = "".join(bit_string_list)
                            bit_string_int_flip = int(bit_string_flip,2)
                            
                            reduced_pvec = np.zeros(2)
                            # if 0->1
                            if bit_string_int < bit_string_int_flip:
                                if bit_string_int in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string_int]
                                if bit_string_int_flip in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string_int_flip]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > threshold:
                                    corrected_counts[bit_string_int] = reduced_pvec_mit[0]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_int] = 0
                                    del corrected_counts[bit_string_int]
                                if abs(reduced_pvec_mit[1]) > threshold:
                                    corrected_counts[bit_string_int_flip] = reduced_pvec_mit[1]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_int_flip] = 0
                                    del corrected_counts[bit_string_int_flip]
                            # if 1->0
                            else:
                                if bit_string_int in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string_int]
                                if bit_string_int_flip in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string_int_flip]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > threshold:
                                    corrected_counts[bit_string_int_flip] = reduced_pvec_mit[0]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_int_flip] = 0
                                    del corrected_counts[bit_string_int_flip]
                                if abs(reduced_pvec_mit[1]) > threshold:
                                    corrected_counts[bit_string_int] = reduced_pvec_mit[1]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_int] = 0
                                    del corrected_counts[bit_string_int]
                corrected_counts = find_closest_counts(corrected_counts, self.shots)
                print(f'{len(corrected_counts)}')
                counts_list_mit[i][operator] = corrected_counts
                
            print(f'rep {i} done')
        return counts_list_mit
        #counts_list_mit = copy.deepcopy(counts_list)
        #for i in range(self.reps):
        #    for basis, counts in counts_list[i].items():
        #        corrected_counts = copy.deepcopy(counts)
        #        for q in mitigate_qubits:
        #            idx = self.qubits_to_connect.index(q)
        #            calibration_M = la.inv(self.M_list[q])
        #            reduced_pvec = np.zeros(2)
                    #for bit_string, count in counts.items():
        #            for bit_string, count in corrected_counts.items():    
        #                if bit_string[::-1][idx] == '0':
        #                    reduced_pvec[0] += count
        #                else:
        #                    reduced_pvec[1] += count
        #            reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
        #            calibration_factor_0 = reduced_pvec_mit[0]/reduced_pvec[0]
        #            calibration_factor_1 = reduced_pvec_mit[1]/reduced_pvec[1]
                    
        #            for bit_string, count in counts.items():
        #                if bit_string[::-1][idx] == '0':
        #                    corrected_counts[bit_string] *= calibration_factor_0
        #                else:
        #                    corrected_counts[bit_string] *= calibration_factor_1
                    #for bit_string, count in corrected_counts.items():
                    #    if count < 0.1:
                    #        corrected_counts[bit_string] = 0
                    #normalise
                    #norm = self.shots/sum(corrected_counts.values())
                    #corrected_counts.update((k,v*norm) for k,v in corrected_counts.items())
                #corrected_counts = find_closest_counts(corrected_counts, self.shots)
        #        counts_list_mit[i][basis] = corrected_counts
                #print(f'{basis} done')
        #    print(f'rep {i} done')
        #return counts_list_mit
    
    def apply_reduced_qrem(self, counts_list, mitigate_qubits = [1,3,4,5,6]):
        """Apply Reduced Quantum Readout Error Mitigation to normal counts

        Args:
            counts_list (list): list of counts dictionaries
            mitigate_qubits (list, optional): list of qubits indicies to mitigate. Defaults to [1,3,4,5,6].
        Returns:
            list: error mitigated counts_list
        """
        counts_list_mit = copy.deepcopy(counts_list)
        for i in range(self.reps):
            for operator, counts in counts_list[i].items():
                corrected_counts = copy.deepcopy(counts)
                #iterate over each qubit
                for q in mitigate_qubits:
                    idx = self.qubits_to_connect.index(q)
                    calibration_M = la.inv(self.M_list[q])
                    applied_names = set([])
                    corrected_bit_strings = [k for k in corrected_counts.keys()]
                    for bit_string in corrected_bit_strings:
                        bit_string_int = int(bit_string, 2)
                        #bit_string = bin(bit_string_int)[2:].zfill(self.nqubits)
                        bit_string_list = list(bit_string[::-1])
                        bit_string_list[idx] = '_'
                        #check if the bit-string (except digit q) is already been corrected
                        name = "".join(bit_string_list)
                        if name not in applied_names:
                            applied_names.add(name)
                            #check the digit is 0 or 1, then flip it
                            if (bit_string_int & (1 << idx)) != 0:
                                bit_string_list[idx] = '0'
                            else:
                                bit_string_list[idx] = '1'
                            bit_string_flip = "".join(bit_string_list)
                            bit_string_int_flip = int(bit_string_flip[::-1],2)
                            
                            reduced_pvec = np.zeros(2)
                            # if 0->1
                            if bit_string_int < bit_string_int_flip:
                                if bit_string in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string]
                                if bit_string_flip[::-1] in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string_flip[::-1]]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > 0.1:
                                    corrected_counts[bit_string] = reduced_pvec_mit[0]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string] = 0
                                    del corrected_counts[bit_string]
                                if abs(reduced_pvec_mit[1]) > 0.1:
                                    corrected_counts[bit_string_flip[::-1]] = reduced_pvec_mit[1]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_flip[::-1]] = 0
                                    del corrected_counts[bit_string_flip[::-1]]
                            # if 1->0
                            else:
                                if bit_string in corrected_counts:
                                    reduced_pvec[1] += corrected_counts[bit_string]
                                if bit_string_flip[::-1] in corrected_counts:
                                    reduced_pvec[0] += corrected_counts[bit_string_flip[::-1]]
                                reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                                if abs(reduced_pvec_mit[0]) > 0.1:
                                    corrected_counts[bit_string_flip[::-1]] = reduced_pvec_mit[0]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string_flip[::-1]] = 0
                                    del corrected_counts[bit_string_flip[::-1]]
                                if abs(reduced_pvec_mit[1]) > 0.1:
                                    corrected_counts[bit_string] = reduced_pvec_mit[1]
                                #zero-out if below threshold
                                else:
                                    corrected_counts[bit_string] = 0
                                    del corrected_counts[bit_string]
                corrected_counts = find_closest_counts(corrected_counts, self.shots)
                #print(f'{operator} done, size {len(corrected_counts)}')
                counts_list_mit[i][operator] = corrected_counts
                
            print(f'rep {i} done')
        return counts_list_mit
    
    def apply_mthree(self, counts_list, mit = None, filename = None):
        """Apply M3Mitigation to counts list

        Args:
            counts_list (list): list of counts dictionaries
            mit : mthreeMitigation object with corresponding backend Defaults to None.
            filename (json file, optional): stored calibration data. Defaults to None.

        Returns:
            list: mitigated counts list
        """
        #Load calibration data
        if mit is None:
            mit = load_cal(self.backend, filename = filename)
        self.mit = mit
        
        counts_list_mit = counts_list.copy()
        
        for i in range(self.reps):
            for operator, counts in counts_list[i].items():
                #Apply correction to counts
                corrected_counts = self.mit.apply_correction(counts, self.qubits_to_connect)
                counts_list_mit[i][operator] = corrected_counts.nearest_probability_distribution()
                print(f'{operator} done')
                #Unnormalise
                for bit_str, prob in counts_list_mit[i][operator].items():
                    counts_list_mit[i][operator][bit_str] = prob*self.shots
            print(f'rep {i} done')
        return counts_list_mit
    
    def calc_moments_from_counts(self, counts_list):
        """Calculate moments (H, H^2, H^3...) from probability vectors

        Args:
            counts_list (list): list of counts dictionaries

        Returns:
            list: list of dictionary of moments at different trials
        """
        #Hamiltonian_values_list = []
        moments_dict = {i+1: [] for i in range(self.order)}
        
        for i in range(self.reps):
            #Hamiltonian_values = copy.deepcopy(self.Hamiltonian_powers)
            expect_val_array = np.zeros(self.order, dtype = complex)
            for hamiltonian_basis in self.Hamiltonian_powers.keys():
                #for each hamiltonian basis, find its corresponding measurement basis, find its expected value
                measurement_basis = self.measurement_mapping[hamiltonian_basis]
                #measurement_string = self.operator_to_string(measurement_basis)
                expect_val_array += self.Hamiltonian_powers[hamiltonian_basis]*self.expected_value_from_counts(counts_list[i][measurement_basis],
                                                                                                               hamiltonian_basis)
            for j in range(len(expect_val_array)):
                moments_dict[j+1].append(expect_val_array[j])
        #moments_dict = {i+1: [] for i in range(self.order)}
        #dictionary where keys are order, values are list of moments on every repetitions
        #for i in range(self.order):
        #    for j in range(self.reps):
        #        Hamiltonian_powers = Hamiltonian_values_list[j]
        #        moment = 0
        #        for array in Hamiltonian_powers.values():
        #            moment += array[i]
        #        moments_dict[i+1].append(moment)
            print(f'rep{i} done')
        self.moments = moments_dict
        return moments_dict
    
    def calc_moments_from_zipped_counts(self, counts_list):
        """Calculate moments (H, H^2, H^3...) from probability vectors
        (from zipped counts)
        Args:
            counts_list (list): list of counts dictionaries

        Returns:
            list: list of dictionary of moments at different trials
        """
        moments_dict = {i+1: [] for i in range(self.order)}
        
        for i in range(self.reps):
            expect_val_array = np.zeros(self.order, dtype = complex)
            for hamiltonian_basis in self.Hamiltonian_powers.keys():
                measurement_basis = self.measurement_mapping[hamiltonian_basis]
                expect_val_array += self.Hamiltonian_powers[hamiltonian_basis]*self.expected_value_from_zipped_counts(counts_list[i][measurement_basis],
                                                                                                                      hamiltonian_basis)
            for j in range(len(expect_val_array)):
                moments_dict[j+1].append(expect_val_array[j])

            print(f'rep{i} done')
        self.moments = moments_dict
        return moments_dict
    
    def calc_moments_from_pvecs(self, pvecs_list):
        """Calculate moments (H, H^2, H^3...) from probability vectors
        Args:
            pvecs_list (list): list of pvecs dictionaries

        Returns:
            list: list of dictionary of moments at different trials
        """
        moments_dict = {i+1: [] for i in range(self.order)}
        #Hamiltonian_values_list = []
        for i in range(self.reps):
            #Hamiltonian_values = copy.deepcopy(self.Hamiltonian_powers)
            expect_val_array = np.zeros(self.order, dtype = complex)
            for hamiltonian_basis in self.Hamiltonian_powers.keys():
                #for each hamiltonian basis, find its corresponding measurement basis, find its expected value
                #hamiltonian_string = self.operator_to_string(hamiltonian_basis)
                measurement_basis = self.measurement_mapping[hamiltonian_basis]
                #measurement_string = self.operator_to_string(measurement_basis)
                expect_val_array += self.Hamiltonian_powers[hamiltonian_basis]*self.expected_value_from_pvecs(pvecs_list[i][measurement_basis],
                                                                                         hamiltonian_basis)
                #Hamiltonian_values[hamiltonian_basis] *= self.expected_value_from_pvecs(pvecs_list[i][measurement_string], 
                #                                                                          hamiltonian_basis) 
            #Hamiltonian_values_list.append(Hamiltonian_values) #Now arrays are expected values*coefficients --> just need to sum
            for j in range(len(expect_val_array)):
                moments_dict[j+1].append(expect_val_array[j])
        
        #dictionary where keys are order, values are list of moments on every repetitions
        #for i in range(self.order):
        #    for j in range(self.reps):
        #        Hamiltonian_powers = Hamiltonian_values_list[j]
        #        moment = 0
        #        for array in Hamiltonian_powers.values():
        #            moment += array[i]
        #        moments_dict[i+1].append(moment)
        
        self.moments = moments_dict
        return moments_dict
    
    def moments_from_result(self, result, apply_mit = 'QREM', order = 4, filename = None, 
                            mitigate_qubits = [1,3,4,5,6], threshold = 0.1):
        """Calculate moments (H, H^2, H^3...) from raw result up to order

        Args:
            result (IBMJob.result() or dict): job result completed
            apply_mit (str, optional): Error mitigation mode. Defaults to 'QREM'.
            order (int, optional): highest order of moments. Defaults to 4.
            filename (json file, optional): M3 calibration data file. Defaults to None.
            mitigate_qubits (list, optional): list of qubits to mitigate in reduced_QREM. Defaults to [1,3,4,5,6].
            threshold (float, optional): zero-out threshold in reduced_QREM. Defaults to 0.1.

        Returns:
            dict: dictionary of moments at different trials and orders
        """
        
        if apply_mit == 'QREM':
            pvecs_list = self.pvecs_from_result(result, order = order)
            pvecs_list = self.apply_qrem(pvecs_list)
            moments_dict = self.calc_moments_from_pvecs(pvecs_list)
        elif apply_mit == 'M3':
            counts_list = self.counts_from_result(result, order = order)
            counts_list = self.apply_mthree(counts_list, filename = filename)
            moments_dict = self.calc_moments_from_counts(counts_list)
        elif apply_mit == 'reduced_QREM':
            counts_list = self.zipped_counts_from_result(result, order = order)
            self.gen_M_list(result)
            counts_list = self.apply_zipped_reduced_qrem(counts_list, mitigate_qubits = mitigate_qubits, threshold = threshold)
            moments_dict = self.calc_moments_from_zipped_counts(counts_list)
            #counts_list = self.counts_from_result(result, order = order)
            #self.gen_M_list(result)
            #counts_list = self.apply_reduced_qrem(counts_list, mitigate_qubits = mitigate_qubits)
            #moments_dict = self.calc_moments_from_counts(counts_list)
        else:
            counts_list = self.counts_from_result(result, order = order)
            moments_dict = self.calc_moments_from_counts(counts_list)
        
        return moments_dict
    
    def cummulants_from_moments(self, moments_dict):
        """convert moments to connected moments (C1=<H>, C2=<H^2>-<H>^2, ...)

        Args:
            moments_dict (dict): moment dictionaries

        Returns:
            dict: dictionary of cummulants
        """
        C = {order+1 : [] for order in range(self.order)}
        for i in range(self.reps):
            M = {order: moments[i] for order, moments in moments_dict.items()}
            for moment in M:
                if abs(moment.imag) >1e-4:
                    print(f'moment of value: {moment} Non-negligible imaginary component')
            for order in range(1, self.order+1):
                c = M[order].copy()
                for p in range(0, order-1):
                    c -= factorial(order-1)/(factorial(p)*factorial(order-1-p))*C[p+1][i]*M[order-1-p]
                C[order].append(c)
        return C
        
    def s_star_from_cummulants(self, C):
        """convert Cummulants to s_star (or z_star, the parameter z when Energy is minimised
        in characteristic equation)

        Args:
            C (dict): Cummulants
        """
        s_star_list = []
        for i in range(self.reps):
            delta = (3*(C[3][i]**2))-(2*C[2][i]*C[4][i])
            #if delta < 0 and abs(delta) > 1e-2*abs(C[1][i]):
            #    sqrt = 0
            if delta <= 0:
                sqrt = 0
                continue
                #warn('Imaginary result')
    #         warn(f'Imaginary result: C={round(C,4)}')
            else:
                sqrt = np.sqrt(delta)
                
            s_star = (C[2][i]**3*(delta-C[3][i]*sqrt))/(delta*(C[3][i]**2-C[2][i]*C[4][i]))
            s_star_list.append(s_star)
            
        return s_star_list
    
    def energy_from_cummualnts(self, C):
        """Calculate perturbed average energy (w.r.t. to the Hamiltonian H) from cummulants

        Args:
            C (dict): cummulants

        Returns:
            list: energies
        """
        ground_state_energies_list = []
        for i in range(self.reps):
            X = 3*C[3][i]**2-2*C[2][i]*C[4][i]
            if X < 0:
                continue
            sqrt = np.sqrt(X)
            E0 = C[1][i]-C[2][i]**2/(C[3][i]**2-C[2][i]*C[4][i])*(sqrt-C[3][i])
            ground_state_energies_list.append(E0)
        return ground_state_energies_list

    def expected_value_from_pvecs(self, pvec, parent_basis):
        """Calculate the expected value from probability vector (each element is counts from bit string.
        Note pauli_string indicates which bit should be taken into account (if I then ignore)

        Args:
            pvec (numpy array): probability vector
            parent_basis (str): the pauli-string of one of the term in Hamiltonian basis (not measurement basis)

        Returns:
            float: expected value of the pvec to the parent basis
        """
        expected_val = 0
        #identity_idx = []
        #Find the indicies where Identity appears in pauli string
        #for i in range(len(pauli_string)):
        #    if pauli_string[i] == 'I':
        #        identity_idx.append(i)
        
        non_identities = parent_basis[0]|parent_basis[1]#Non-identity digits
            
        #Calculate the expected value of the pauli string measurement <pauli_string>
        for idx in range(len(pvec)):
            expected_val += pvec[idx]*(-1)**hamming_weight(non_identities&idx)#only count '1's from non-identity digits
            
            #one_count = 0 #count number of 1s appeared in a specific bit string measured
                          # if odd 1s, negative contribution (product = -1), if even 1s, positive contribution (product = +1)
            #for i in range(len(bit_str)):
                #count number of 1s while ignore if it was measured from Identity
            #    if bit_str[i] == '1' and not (i in identity_idx):
            #        one_count += 1
            #if one_count % 2 == 0:
            #    expected_val += pvec[idx]
            #else:
            #    expected_val -= pvec[idx]
        return expected_val
    
    
    def expected_value_from_counts(self, counts, parent_basis):
        """Calculate the expected value from counts (each element is counts from bit string.
        Note pauli_string indicates which bit should be taken into account (if I then ignore)

        Args:
            counts (dict): counts of the measurement basis(circuit)
            parent_basis (str): the pauli-string of one of the term in Hamiltonian basis (not measurement basis)

        Returns:
            float: expected value of the counts to the parent basis
        """
        expected_val = 0
        non_identities = parent_basis[0]|parent_basis[1]#Non-identity digits
        for bit_str, count in counts.items():
            expected_val += count*(-1)**hamming_weight(non_identities&int(bit_str[::-1],2))#only count '1's from non-identity digits
        expected_val /= self.shots
        
        return expected_val
    
    def expected_value_from_zipped_counts(self, counts, parent_basis):
        """Calculate the expected value from counts (each element is counts from bit string.
        Note pauli_string indicates which bit should be taken into account (if I then ignore)
        (from zipped counts)
        Args:
            counts (dict): counts of the measurement basis(circuit)
            parent_basis (str): the pauli-string of one of the term in Hamiltonian basis (not measurement basis)

        Returns:
            float: expected value of the counts to the parent basis
        """
        expected_val = 0
        non_identities = parent_basis[0]|parent_basis[1]#Non-identity digits
        for bit_str_int, count in counts.items():
            expected_val += count*(-1)**hamming_weight(non_identities&bit_str_int)#only count '1's from non-identity digits
        expected_val /= self.shots
        
        return expected_val
    
    def check_BWC(self, operator_a, operator_b):
        """check whether operator a and b in integers are bitwise-commuting

        Args:
            operator_a (tuple): operator A in (X,Z) language
            operator_b (tuple): operator B in (X,Z) language

        Returns:
            bool: True if A B are BWC, False if not
        """
        x1,z1 = operator_a
        x2,z2 = operator_b
        #return not ((x1&z2) or (z1&x2) or (y1&(x3^z3)))
        return not ((x1|z1) & (x2|z2) & ((x1^x2)|(z1^z2)))
    
    def is_measured_by(self, operator_a, operator_b):
        """check if 

        Args:
            operator_a (tuple): operator A in (X,Z) language
            operator_b (tuple): operator B in (X,Z) language

        Returns:
            bool: True if A is measured by B (has less identities and BWC)
        """
        if not self.check_BWC(operator_a, operator_b):
            return False
        x1,z1 = operator_a
        x2,z2 = operator_b
        
        return (x1&x2==x1 and z1&z2==z1)
    
    def operator_to_string(self, operator):
        """Convert an operator from integer tuple to readable string

        Args:
            operator (tuple): operator in (X,Z) language

        Returns:
            str: pauli-string of this operator
        """
        x,z = operator
        out = max(x,z)
        if not out:
            return 'I'*self.nqubits
        
        sz=int(log2(float(out)))+1
        #Note Integer reads large to small from L to R in binary, so need to reverse the pauli string
        pauli_string = ''.join(P2S_LOOKUP[(bool(x&2**ix),bool(z&2**ix))] for ix in range(sz)).ljust(self.nqubits,'I')[::-1]
        return pauli_string


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