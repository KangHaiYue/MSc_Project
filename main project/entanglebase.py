# -*- coding: utf-8 -*-
"""
Last Modified on 19 August, 2023
@author: Haiyue Kang (Modified from John. F. Kam)
@Link to John. F. Kam's original codes: https://github.com/jkfids/qiskit-entangle/blob/main/code/entanglebase.py
"""

# Standard libraries
import networkx as nx


class EntangleBase:
    """Parent class for device entangled state analysis"""
    
    def __init__(self, backend):
        self.backend = backend
        
        self.device_name = backend.properties().backend_name
        self.nqubits = len(backend.properties().qubits)
        self.graph, self.connections, self.edge_params = self.gen_graph()
        self.edge_list = sorted(list(self.edge_params.keys()), key=lambda q: (q[0], q[1]))
        self.nedges = len(self.edge_params)
    
    def gen_graph(self):
        """
        Obtain the Graph of the IBM QPU, including connections (neighbour of each qubit) and edges
        """
        graph = nx.Graph()
        connections = {} #{0:[1], 1:[2], 2:[3,1],...} key is the index of the qubit, value is/are the qubit(s) connected to it
        edges = {} #{(0,1):0.2, (1,2):0.5,...} the values are the CNOT errors for this pair of qubits
        for i in range(self.nqubits):
            connections[i] = []
        # Iterate over possible cnot connections
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
            #if gate.gate == 'ecr':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                #q0 = sorted(gate.qubits)[0]
                #q1 = sorted(gate.qubits)[1]#originally there is no sorted in Fidel's code, adding this to account for non-cx gates
                connections[q0].append(q1)
                if q0 < q1:
                    #graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                    graph.add_edge(q0, q1, weight=1)
                    edges[q0, q1] = gate.parameters[0].value
        # Sort adjacent qubit list in ascending order
        for q in connections:
            connections[q].sort()
            
        return graph, connections, edges