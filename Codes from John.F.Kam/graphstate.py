# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:01:06 2022

@author: Fidel
"""

# Standard libraries
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import networkx as nx

# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.providers.aer.noise import NoiseModel

# Local modules
from utilities import pauli_n, bit_str_list
from entanglebase import EntangleBase


# Two-qubit Pauli basis
basis_list = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                  'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ',
                  'ZI', 'ZX', 'ZY', 'ZZ']


class GraphState(EntangleBase):
    """
    Class to run native-graph state negativity measurement experiments

    """

    def __init__(self, backend):
        """
        Initialize from EntangleBase parent class and additionally obtain
        edge adjacencies and generate the native-graph state preperation
        circuit

        """
        super().__init__(backend)  # Inherent from parent class

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
        Generate a native-graph state circuit over every physical edge

        """

        circ = QuantumCircuit(self.nqubits)
        unconnected_edges = self.edge_list.copy()
        depths = []
        # Apply Hadamard gates to every qubit
        circ.h(self.qubits)
        # Connect every edge with cz gates
        while unconnected_edges:
            connected_qubits = []  # Qubits already connected in the current time step
            #connected_edges = []
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

    def run_qst(self, reps=1, shots=4096, delay=None, dd=None, pulserate=4, 
                qrem=False, sim=None, output='default', execute_only=False):
        """
        Run entire QST program to obtain qubit pair density matrices with
        option to only send job request

        """

        self.gen_qst_circuits(delay, dd, pulserate)
        job = self.run_qst_circuits(reps, shots, qrem, sim)

        if execute_only is True:  # If only executing job
            return job

        # Otherwise obtain jobs results
        result = job.result()
        rho_dict = self.qst_from_result(result, output)
        return rho_dict

    def qst_from_result(self, result, output='default'):
        """
        Process Qiskit Result into qubit pair density matrices. Can be used
        with externally obtained results.

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
        """
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

            self.batches = batches
            self.group_list = group_list

        return batches

    def gen_qst_circuits(self, delay=None, dd=None, pulserate=4):
        """
        Generates (parallelised) quantum state tomography circuits

        """
        # Generate batches of groups (target edges + adjacent qubits) to perform
        # QST in parallel
        if self.batches is None:
            self.batches = self.gen_batches()

        circuits = {}  # Dictionary of groups of circuits where batches are keys
        name_list = []  # List of circuit names
        
        graphstate = self.circuit.copy()
        graphstate.barrier()
        
        if delay is not None:
            fdelay = self.format_delays(delay)
            if dd is None:
                graphstate.delay(fdelay)
            elif dd == 'hahn':
                tdelay = fdelay - 2*self.tx
                fqdelay = self.format_delays(tdelay/4, unit='dt')
                graphstate.delay(fqdelay)
                graphstate.x(self.qubits)
                graphstate.delay(2*fqdelay)
                graphstate.x(self.qubits)
                graphstate.delay(fqdelay)
            elif dd == 'pdd':
                pulses = int(0.5*pulserate*delay)*2
                dt = self.format_delays((fdelay - pulses*self.tx)/pulses, unit='dt')
                padding = self.format_delays(dt/2, unit='dt')
                
                graphstate.delay(padding)
                graphstate.x(self.qubits)
                for i in range(pulses - 1):
                    graphstate.delay(dt)
                    graphstate.x(self.qubits)
                graphstate.delay(padding)
        

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

            circuits[batch] = batch_circuits

            self.qst_circuits = circuits
            self.name_list = name_list

        return circuits

    def run_qst_circuits(self, reps=1, shots=4096, qrem=False, sim=None):
        """
        Execute the quantum state tomography circuits

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
                          initial_layout=self.qubits,
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
        Get counts from qiskit result as dictionary or lists of dictionaries

        """

        if self.reps is None:
            self.reps = int(len(result.results)/len(self.name_list))
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
                qst_counts[batch][basis] = result.get_counts(name_ext)
            qst_counts_multi.append(qst_counts)

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

        if self.reps == 1:
            return qst_counts  # Single experiment

        return qst_counts_multi  # Multiple experiments

    def group_counts(self, qst_counts):
        """
        Regroups qst_counts according to tomography groups (target qubit pair
        + adjacent qubits) and convert into equivalent probability vector

        """

        g_counts_list = []  # Regrouped counts
        g_vecs_list = []  # Equivalent probability vectors

        if isinstance(qst_counts, dict):
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
                g_vecs[group] = {basis: np.zeros(2**n) for basis in basis_list}

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
                            g_counts[group][basis][split[ii]] += count
                            g_vecs[group][basis][int(split[ii], 2)] += count

            g_counts_list.append(g_counts)
            g_vecs_list.append(g_vecs)

        if self.reps == 1:
            return g_counts, g_vecs  # Single experiment

        return g_counts_list, g_vecs_list  # Multiple experiments

    def bin_pvecs(self, g_counts):
        """
        Further classify the group probability vectors according to the different
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

            for group, basis_counts in g_counts[i].items():
                edge = group[:2]
                for basis, counts in basis_counts.items():
                    for bit_str, count in counts.items():
                        idx = int(bit_str[:2], 2)
                        bn = bit_str[2:]

                        b_pvecs[edge][bn][basis][idx] += count

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
        Reconstruct the density matrices for every qubit pair for every
        measurement combination and save it as a dictionary or list of
        dictionaries

        """

        rho_dict_list = []

        qst_counts = self.counts_from_result(result)
        g_counts, g_vecs = self.group_counts(qst_counts)

        # Whether mitigation is applied or not defaults to whether qrem circuits
        # are included in result
        if apply_mit is None:
            apply_mit = self.qrem
        # If mitigation is applied
        if apply_mit is True:
            g_counts, g_vecs = self.apply_qrem(g_counts, g_vecs)

        b_pvecs = self.bin_pvecs(g_counts)

        if isinstance(b_pvecs, dict):
            b_pvecs = [b_pvecs]

        for i in range(self.reps):
            rho_dict = {edge: {} for edge in self.edge_list}
            for edge, bns in b_pvecs[i].items():
                for bn, pvecs in bns.items():
                    rho_dict[edge][bn] = GraphState.calc_rho(pvecs)
            rho_dict_list.append(rho_dict)

        if self.reps == 1:
            return rho_dict

        return rho_dict_list

    def gen_qrem_circuits(self):
        """
        Generate QREM circuits

        """

        circ0 = QuantumCircuit(self.nqubits, self.nqubits, name='qrem0')
        circ0.measure(self.qubits, self.qubits)

        circ1 = QuantumCircuit(self.nqubits, self.nqubits, name='qrem1')
        circ1.x(self.qubits)
        circ1.measure(self.qubits, self.qubits)

        self.qrem_circuits = [circ0, circ1]

        return [circ0, circ1]

    def apply_qrem(self, g_counts, g_vecs):
        """
        Apply quantum readout error mitigation on grouped counts/probability
        vectors

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

        return g_counts_list, g_vecs_list

    def calc_M_multi(self, qubits):
        """
        Compose n-qubit calibration matrix by tensoring single-qubit matrices

        """

        M = self.M_list[qubits[0]]
        for q in qubits[1:]:
            M_new = np.kron(M, self.M_list[q])
            M = M_new

        return M

    @staticmethod
    def calc_rho(pvecs):
        """
        Calculate density matrix from probability vectors

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
            s_dict['I' + basis[1]] += (pvec[0] - pvec[1] + pvec[2] - pvec[3])/3
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
        """
        ALgorithm to find closest physical density matrix from Smolin et al.

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
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            # Step 5 Construct new density matrix
            rho_physical += eigval_new[j] * \
                np.outer(eigvec[:, j], eigvec[:, j].conjugate())

        return rho_physical
    
    def format_delays(self, delays, unit='us'):

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
            scale = 1

        # Qiskit only accepts multiples of 16*dt
        if n is None:
            # For single time
            delays_new = np.floor(delays*scale/16)*16
        else:
            # For array of times
            delays_new = np.zeros(n)
            for i, t in enumerate(delays):
                delays_new[i] = np.round(t*scale/16)*16

        return delays_new


def calc_negativities(rho_dict, mode='all'):
    """
    Obtain negativities corresponding to every density matrix in rho_dict.
    Option to obtain max, mean or all negativities between measurement
    combinations (bins)

    """

    n_all_list = []  # Negativities for each bin per experiment
    n_mean_list = []  # Mean negativity between bins per experiment
    n_max_list = []  # Max negativity between bins per experiment
    n_min_list = []

    if isinstance(rho_dict, dict):
        rho_dict = [rho_dict]

    nexp = len(rho_dict)

    for i in range(nexp):
        n_all = {edge: {} for edge in rho_dict[i].keys()}
        n_mean = {}
        n_max = {}
        n_min = {}

        for edge, bns in rho_dict[i].items():
            n_sum = 0.
            n_list = []

            for bn, rho in bns.items():
                n = calc_n(rho)

                n_all[edge][bn] = n
                n_sum += n
                n_list.append(n)

            n_mean[edge] = n_sum/len(bns)
            n_max[edge] = max(n_all[edge].values())
            n_min[edge] = min(n_all[edge].values())

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


def calc_n(rho):
    """
    Calculate the negativity of bipartite entanglement for a given 2-qubit
    density matrix

    """

    rho_pt = ptrans(rho)
    w, _ = la.eig(rho_pt)
    n = np.sum(w[w < 0])

    return abs(n)

def ptrans(rho):
    """Obtain the partial transpose of a 4x4 array (A kron B) w.r.t B"""

    rho_pt = np.zeros(rho.shape, dtype=complex)
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            rho_pt[i:i+2, j:j+2] = rho[i:i+2, j:j+2].transpose()

    return rho_pt


def calc_n_mean(n_list):
    """Calculate mean negativity dict from lists of negativity dicts"""

    edges = n_list[0].keys()
    N = len(n_list)

    n_dict = {edge: [n_list[i][edge] for i in range(N)] for edge in edges}
    n_mean = {key: np.mean(value) for key, value in n_dict.items()}
    n_std = {key: np.std(value)/np.sqrt(len(value)) for key, value in n_dict.items()}

    return n_mean, n_std



def get_negativity_info(n_list, nmit_list):
    
    n_mean_dict, _ = calc_n_mean(n_list)
    n_mean = np.mean(list(n_mean_dict.values()))
    std = np.std(list(n_mean_dict.values()))

    nmit_mean_dict, _ = calc_n_mean(nmit_list)
    nmit_mean = np.mean(list(nmit_mean_dict.values()))
    std_mit = np.std(list(nmit_mean_dict.values()))
    
    n5 = get_largest_connected(n_mean_dict, threshold=0.025)
    n50 = get_largest_connected(n_mean_dict, threshold=0.25)
    n75 = get_largest_connected(n_mean_dict, threshold=0.75*0.5)
    n90 = get_largest_connected(n_mean_dict, threshold=0.9*0.5)
    
    nmit5 = get_largest_connected(nmit_mean_dict, threshold=0.025)
    nmit50 = get_largest_connected(nmit_mean_dict, threshold=0.25)
    nmit75 = get_largest_connected(nmit_mean_dict, threshold=0.75*0.5)
    nmit90 = get_largest_connected(nmit_mean_dict, threshold=0.9*0.5)
    
    info_dict = {'Mean negativity': n_mean,
                 'std': std,
                'Mean negativity (mit)': nmit_mean,
                 'std (mit)': std_mit,
                'Connected N>5%': len(n5),
                'Connected N>50%': len(n50),
                'Connected N>75%': len(n75),
                'Connected N>90%': len(n90),
                'Connected N>5% (mit)': len(nmit5),
                'Connected N>50% (mit)': len(nmit50),
                'Connected N>75% (mit)': len(nmit75),
                'Connected N>90% (mit)': len(nmit90)}

    return info_dict


def get_largest_connected(n_dict, threshold=0.25):
    
    edges = filter_edges(n_dict, threshold)
    G = nx.Graph()
    G.add_edges_from(edges)
    try:
        largest = max(nx.connected_components(G), key=len)
    except:
        largest = {}
    
    return largest

    
def filter_edges(n_dict, threshold=0.25/2):
    return [key for key, value in n_dict.items() if value >= threshold]


def get_mean_cnot(graphstate, properties):
    err_array = np.zeros(len(graphstate.edge_list))
    for i, edge in enumerate(graphstate.edge_list):
        err = properties.gate_error('cx', edge)
        if err < 0.9:
            err_array[i] = err
        #print(edge, err_array[i])
    
    return np.mean(err_array), np.std(err_array)

