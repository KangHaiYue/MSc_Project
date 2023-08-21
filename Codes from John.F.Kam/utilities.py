# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:37:39 2021

@author: Fidel
"""

# Standard libraries
from numpy import array, kron

# Qiskit libraries
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
import mthree

#from token import tokenid
from tokenid import tokenid 

#try:
#    import token.token
#except:
    #token = 
#    pass

# IBM Quantum account utils

#def startup(check=False, token=tokenid, hub='ibm-q-melbourne', group=None, project=None):
#    

def startup(check=False, token=tokenid, hub='ibm-q-melbourne', group=None, project=None):
    """Start up session"""
    
    try:
        provider = IBMProvider()
    except:
        IBMProvider.save_account(token=tokenid)
        provider = IBMProvider()
    print("Account enabled")
    
    if check:
        check_provider(hub)
            
    return provider
        
def check_provider(hub='ibm-q-melbourne'):
    """Check list of providers with queue size and qubit count for input hub"""
    provider = IBMProvider()
    
    for backend in provider.backends():
      try:
        qubit_count = len(backend.properties().qubits)
      except:
        qubit_count = 'simulated'
      print(f'{backend.name()} has {backend.status().pending_jobs} queud and {qubit_count} qubits')
      
      
# Math objects

pauli = {'I': array([[1, 0], [0, 1]], dtype=complex),
         'X': array([[0, 1], [1, 0]], dtype=complex),
         'Y': array([[0, -1j], [1j, 0]], dtype=complex),
         'Z': array([[1, 0], [0, -1]], dtype=complex)}


# Math functions

def bit_str_list(n):
    """Create list of all n-bit binary strings"""
    return [format(i, 'b').zfill(n) for i in range(2**n)]
      
def pauli_n(basis_str):
    """Calculate kronecker tensor product sum of basis from basis string"""
    
    M = pauli[basis_str[0]]
    try:
        for basis in basis_str[1:]:
            M_new = kron(M, pauli[basis])
            M = M_new
    except: pass # Single basis case
    
    return M 

# Run and load mthree calibrations
def run_cal(backend, initial_layout, filename=None):
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(initial_layout, shots=8192)
    if filename is None:
        filename = f'calibrations/{backend.name()}_cal.json'
    mit.cals_to_file(filename)
    
    return mit
    
def load_cal(backend=None, filename=None):
    mit = mthree.M3Mitigation()
    if filename is None:
        filename = f'calibrations/{backend.name()}_cal.json'
    mit.cals_from_file(filename)
    
    return mit

def list_retrieve_jobs(job_ids):
    for i, job_id in enumerate(job_ids):
        print(f'job{i+1} = provider.backend.retrieve_job(\'{job_id}\')')
      

    