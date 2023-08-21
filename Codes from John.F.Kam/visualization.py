# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:36:23 2023

@author: jfide
"""

# Standard
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr
import matplotlib as mpl
import networkx as nx

# Local
from graphstate import GraphState, calc_n_mean, filter_edges
from qiskit.visualization import plot_gate_map, plot_coupling_map
from qubit_coords import qubit_coords127

# GRAPH STATE

def plot_negativities_multi(backend, n_list, nmit_list=None, figsize=(6.4, 4.8), idx=None, return_idx=False, print_n=True):
    """
    Plot average negativity across multiple experiments with error bars as std

    """

    # Figure
    fig, ax = plt.subplots(figsize=figsize)


    # Extract the mean negativity and its standard deviation
    while True:
        try:
            edges = n_list[0].keys()
            n_mean, n_std = calc_n_mean(n_list)
        except:
            n_list = [n_list]
            if nmit_list is not None:
                nmit_list = [nmit_list]
        else:
            break

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
        if idx is None:
            idx = Y1min.argsort()
        Y1 = Y1[idx]
        Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        if idx is None:
            idx = Y0min.argsort()

    X = X[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]

    # Plot
    if print_n is True:
        ax.errorbar(X, Y0, yerr=Y0err, capsize=3, fmt='.', c='r', 
                    label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
        try:
            ax.errorbar(X, Y1, yerr=Y1err, capsize=3, fmt='.', c='b', 
                        label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
        except:
            pass
    else:
        ax.errorbar(X, Y0, yerr=Y0err, capsize=3, fmt='.', c='r', 
                    label='No QREM')
        try:
            ax.errorbar(X, Y1, yerr=Y1err, capsize=3, fmt='.', c='b', 
                        label='QREM')
        except:
            pass

    # Fig params
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.set_ylim([-0.02, 0.52])
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    ax.legend()

    ax.set_xlabel("Qubit Pairs")
    ax.set_ylabel("Negativity")
    #ax.set_title(f"Native-graph state negativities ({backend.name})")
    #ax.set_title(backend.name)
    fig.set_tight_layout(True)

    if return_idx:
        return fig, idx
    else:
        return fig

def plot_negativities127(backend, n_list, nmit_list, figsize=(14, 9), idx=None, return_idx=False):
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)
    
    # Extract the mean negativity and its standard deviation
    try:
        n_list[0]
    except:
        n_list = [n_list]
        nmit_list = [nmit_list]
    edges = n_list[0].keys()
    n_mean, n_std = calc_n_mean(n_list)
    nmit_mean, nmit_std = calc_n_mean(nmit_list)
    
    # Convert into array for plotting
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in edges])
    Y0 = np.fromiter(n_mean.values(), float)
    Y0err = np.fromiter(n_std.values(), float)
    
    Y1 = np.fromiter(nmit_mean.values(), float)
    Y1err = np.fromiter(nmit_std.values(), float)
    
    # Order in increasing minimum negativity (QREM)
    Y1min = Y1 - Y1err
    if idx is None:
        idx = Y1min.argsort()
    Y1 = Y1[idx]
    Y1err = Y1err[idx]
    
    X = X[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]
    
    hp = int(len(X)/2)
    
    ax1.errorbar(X[:hp], Y0[:hp], yerr=Y0err[:hp], capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax1.errorbar(X[:hp], Y1[:hp], yerr=Y1err[:hp], capsize=3, fmt='.', c='b', 
                label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    
    ax2.errorbar(X[hp:], Y0[hp:], yerr=Y0err[hp:], capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax2.errorbar(X[hp:], Y1[hp:], yerr=Y1err[hp:], capsize=3, fmt='.', c='b', 
                label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    
    for ax in (ax1, ax2):
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.set_ylim([-0.02, 0.52])
        ax.tick_params(axis='x', labelrotation=90)
        ax.grid()
        ax.set_ylabel("Negativity")
        ax.margins(0.025, 0.05)
        
    ax1.legend()
    ax2.set_xlabel("Qubit Pairs")
    #ax1.set_title(backend.name)
    fig.set_tight_layout(True)
    
    if return_idx is True:
        fig = fig, idx
    
    return fig

def plot_negativities433(backend, adj_qubits, readout_errors, n_list, nmit_list, figsize=(14, 14), idx=None, return_idx=False):
    
    # Figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=figsize)

    fig2, scatter_plot = plt.subplots(figsize = (10,6))
    # Extract the mean negativity and its standard deviation
    try:
        n_list[0]
    except:
        n_list = [n_list]
        nmit_list = [nmit_list]
    edges = n_list[0].keys()
    n_mean, n_std = calc_n_mean(n_list)
    nmit_mean, nmit_std = calc_n_mean(nmit_list)
    
    # Convert into array for plotting
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in edges])
    
    #
    errors_dict = {}
    #errors_std_dict = {}
    for edge in edges:
        connected_qubits = adj_qubits[edge]
        connected_qubits.append(list(edge)[0])
        connected_qubits.append(list(edge)[1])
        errors = []
        for qubit in connected_qubits:
            errors.append(readout_errors[qubit])
        errors_dict[edge] = 1 - np.prod(np.ones(len(errors))-errors)
        #errors_std_dict[edge] = np.std(errors)/np.sqrt(len(connected_qubits))
        
    Y0 = np.fromiter(n_mean.values(), float)
    Y0err = np.fromiter(n_std.values(), float)
    
    Y1 = np.fromiter(nmit_mean.values(), float)
    Y1err = np.fromiter(nmit_std.values(), float)
    
    read_errors = np.fromiter(errors_dict.values(), float)
    #read_errors_err = np.fromiter(errors_std_dict.values(), float)
    # Order in increasing minimum negativity (QREM)
    
    Y1min = Y1 - Y1err
    if idx is None:
        idx = Y1min.argsort()
    X = X[idx]
    Y1 = Y1[idx]
    Y1err = Y1err[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]
    read_errors = read_errors[idx]
    #read_errors_err = read_errors_err[idx]
    
    qp = int(len(X)/4)
    
    ax1.errorbar(X[:qp], Y0[:qp], yerr=Y0err[:qp], capsize=3, fmt='.', c='r', 
                 label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax1.errorbar(X[:qp], Y1[:qp], yerr=Y1err[:qp], capsize=3, fmt='.', c='b', 
                 label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    #ax1_twin = ax1.twiny()
    #ax1_twin.errorbar(X[:qp], read_errors[:qp], yerr=read_errors_err[:qp], capsize=3, fmt='.', c='g', 
    #                  label=f'Average readout assignment error: {np.mean(read_errors):.3f}')
    
    ax2.errorbar(X[qp:qp*2], Y0[qp:qp*2], yerr=Y0err[qp:qp*2], capsize=3, fmt='.', c='r', 
                 label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax2.errorbar(X[qp:qp*2], Y1[qp:qp*2], yerr=Y1err[qp:qp*2], capsize=3, fmt='.', c='b', 
                 label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    #ax2_twin = ax2.twiny()
    #ax2_twin.errorbar(X[qp:qp*2], read_errors[qp:qp*2], yerr=read_errors_err[qp:qp*2], capsize=3, fmt='.', c='g', 
    #                  label=f'Average readout assignment error: {np.mean(read_errors):.3f}')
    
    ax3.errorbar(X[qp*2:qp*3], Y0[qp*2:qp*3], yerr=Y0err[qp*2:qp*3], capsize=3, fmt='.', c='r', 
                 label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax3.errorbar(X[qp*2:qp*3], Y1[qp*2:qp*3], yerr=Y1err[qp*2:qp*3], capsize=3, fmt='.', c='b', 
                 label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    #ax3_twin = ax3.twiny()
    #ax3_twin.errorbar(X[qp*2:qp*3], read_errors[qp*2:qp*3], yerr=read_errors_err[qp*2:qp*3], capsize=3, fmt='.', c='g', 
    #                  label=f'Average readout assignment error: {np.mean(read_errors):.3f}')
    
    ax4.errorbar(X[qp*3:qp*4], Y0[qp*3:qp*4], yerr=Y0err[qp*3:qp*4], capsize=3, fmt='.', c='r', 
                 label=f'No QREM (Mean negativity: {np.mean(Y0):.3f})')
    
    ax4.errorbar(X[qp*3:qp*4], Y1[qp*3:qp*4], yerr=Y1err[qp*3:qp*4], capsize=3, fmt='.', c='b', 
                 label=f'QREM (Mean negativity: {np.mean(Y1):.3f})')
    #ax4_twin = ax4.twiny()
    #ax4_twin.errorbar(X[qp*3:qp*4], read_errors[qp*3:qp*4], yerr=read_errors_err[qp*3:qp*4], capsize=3, fmt='.', c='g', 
    #                  label=f'Average readout assignment error: {np.mean(read_errors):.3f}')
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.set_ylim([-0.02, 0.52])
        ax.tick_params(axis='x', labelrotation=90)
        ax.grid()
        ax.set_ylabel("Negativity")
        ax.margins(0.025, 0.05)
    
    mitigation_size = Y1 - Y0
    scatter_plot.scatter(read_errors, mitigation_size, c='r', marker = '.')
    
    m, c = polyfit(read_errors, mitigation_size, 1)
    Xfit = np.linspace(min(read_errors), max(read_errors), 100)
    correlation,_ = pearsonr(read_errors, mitigation_size)
    scatter_plot.plot(Xfit, m*Xfit+c, ls='-', label=f'correlation = {correlation:.3f}')
    #scatter_plot.set_yticks(np.arange(0, 0.55, 0.05))
    scatter_plot.set_ylim([-0.02, 0.52])
    #scatter_plot.set_xticks(np.arange(0, 1, 0.05))
    scatter_plot.set_xlim([-0.02, 1])
    scatter_plot.set_ylabel("mitigation size")
    scatter_plot.set_xlabel('Net readout assignment errors')
    scatter_plot.margins(0.025, 0.05)
    #for ax in (ax1_twin, ax2_twin, ax3_twin, ax4_twin):
    #    ax.set_yticks(np.arange(0, 1, 0.05))
    #    ax.set_ylim([-0.02, 1])
    #    ax.tick_params(axis='x', labelrotation=90)
    #    ax.grid()
    #    ax.set_ylabel("Average Readout assignment error")
    #    ax.margins(0.025, 0.05)
        
    ax1.legend()
    scatter_plot.legend()
    plt.show()
    #ax1_twin.legend()
    ax4.set_xlabel("Qubit Pairs")
    #ax1.set_title(backend.name)
    fig.set_tight_layout(True)
    fig2.set_tight_layout(True)
    if return_idx is True:
        fig = fig, idx
    
    return fig, fig2

def plot_nmap127(graphstate, n_list):
    
    nqubits = 127
    
    n_mean, n_std = calc_n_mean(n_list)
    
    cmap = mpl.cm.get_cmap('RdBu')
    cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName",['r', 'm', 'b'])
    
    edges = filter_edges(n_mean, threshold=0.025)
    G = nx.Graph()
    G.add_edges_from(edges)
    unconnected = list(set(range(nqubits)) - list(nx.connected_components(G))[0])
    
    qubit_n = np.zeros(nqubits)
    qubit_color = []
    edge_list = []
    line_color = []
    
    for key, values in n_mean.items():
        edge_list.append(key)
        line_color.append(mpl.colors.to_hex(cmap(2*values), keep_alpha=True))
        qubit_n[key[0]] += values
        qubit_n[key[1]] += values
        
    for i, n in enumerate(qubit_n):
        x = 2*n/graphstate.graph.degree[i]
        if i in unconnected:
            #qubit_color.append('#D3D3D3')
            qubit_color.append('#C0C0C0')
        else:
            qubit_color.append(mpl.colors.to_hex(cmap(x), keep_alpha=True))
            
    fig = plot_coupling_map(nqubits, qubit_coords127, edge_list, line_color=line_color, qubit_color=qubit_color, \
                            line_width=6, figsize=(12,12))
    
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)

    #ax = fig.get_axes()[0]
    cax = fig.add_axes([0.9, 0.2, 0.015, 0.605])
    
    im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Negativity')
    fig.savefig('output/ghznmap127.png', dpi=400)
    
    return fig
    


def plot_device_nbatches(provider, size=(6.4, 4.8)):
    """Plot the number of QST patches for each available device"""

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


def plot_cxerr_corr(properties, adj_edges, n_mean, inc_adj=True, figsize=(6.4, 4.8)):
    """Plot negativity vs. CNOT error"""

    # Figure
    #fig, ax = plt.subplots(figsize=figsize)

    edges = n_mean.keys()
    
    if inc_adj is True:
        X = []
        for edge in edges:
            targ_err = [properties.gate_error('cx', edge)]
            adj_errs = [properties.gate_error('cx', adj_edge) 
                       for adj_edge in adj_edges[edge]
                       if adj_edge in edges]
            err = np.mean(targ_err + adj_errs)
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


# GHZ STATE