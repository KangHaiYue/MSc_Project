# -*- coding: utf-8 -*-
"""
Created on Thu May 12 00:11:10 2022

@author: Fidel
"""

# Standard libraries
import networkx as nx
from qiskit.transpiler import InstructionDurations


class EntangleBase:
    """Parent class for device entangled state analysis"""
    
    def __init__(self, backend):
        self.backend = backend
        
        properties = backend.properties()
        
        self.device_name = properties.backend_name
        self.nqubits = len(properties.qubits)
        self.qubits = list(range(self.nqubits))
        self.graph, self.connections, self.edge_params = self.gen_graph()
        self.edge_list = sorted(list(self.edge_params.keys()), key=lambda q: (q[0], q[1]))
        self.nedges = len(self.edge_params)
        # If there are faulty qubits
        properties = properties
        faulty_qubits = properties.faulty_qubits()
        faulty_gates = properties.faulty_gates()
        faulty_edges = [tuple(gate.qubits) for gate in faulty_gates if len(gate.qubits) > 1
                        ]
        for q in faulty_qubits:
            self.qubits.remove(q)
        for edge in faulty_edges:
            self.edge_list.remove(edge)
        
        durations = InstructionDurations.from_backend(backend)
        self.tx = durations.get('x', 0)
    
    def gen_graph(self):
        """
        """
        graph = nx.Graph()
        connections = {}
        edges = {}
        for i in range(self.nqubits):
            connections[i] = []
        # Iterate over possible cnot connections
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                connections[q0].append(q1)
                if q0 < q1:
                    graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                    edges[q0, q1] = gate.parameters[0].value
            if gate.gate == 'ecr':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                connections[q0].append(q1)
                connections[q1].append(q0)
                if q0 < q1:
                    graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                    edges[q0, q1] = gate.parameters[0].value
                if q1 < q0:
                    graph.add_edge(q1, q0, weight=gate.parameters[0].value)
                    edges[q1, q0] = gate.parameters[0].value
        # Sort adjacent qubit list in ascending order
        for q in connections:
            connections[q].sort()
            
        return graph, connections, edges