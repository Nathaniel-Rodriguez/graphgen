import networkx as nx
import os
import sys
import shutil
import numpy as np
import utilities

def write_flagfile(network_params):
    """
    Generates an LFR benchmark parameter file from chosen network parameters
    """

    # Make lines for each parameter
    lines = ""
    lines += '-N ' + str(network_params['N']) + '\n'
    lines += '-k ' + str(network_params['k']) + '\n'
    lines += '-maxk ' + str(network_params['maxk']) + '\n'
    lines += '-mu ' + str(network_params['mu']) + '\n'
    lines += '-t1 ' + str(network_params['t1']) + '\n'
    lines += '-t2 ' + str(network_params['t2']) + '\n'
    lines += '-minc ' + str(network_params['minc']) + '\n'
    lines += '-maxc ' + str(network_params['maxc']) + '\n'
    lines += '-on ' + str(network_params['on']) + '\n'
    lines += '-om ' + str(network_params['om']) + '\n'

    # Write-out to file
    flag_file = open(network_params['param_file'], 'w')
    flag_file.write(lines)
    flag_file.close()

def generate_graph(network_params, command_file, path):
    """
    Generates a graph from the LFR benchmark program with desired parameters
    """

    # Create parameter file
    write_flagfile(network_params)

    # Run program
    command = command_file + " -f " + network_params['param_file']
    os.popen(command)

    # Read output into networkx graph and return it
    LFR_graph = read_LFR_output(path + "network.dat", path + "community.dat")

    return LFR_graph

def read_LFR_output(edge_file, community_file):
    """
    Reads a LFR style output file into a networkx graph object
    """

    col1, col2, col3 = utilities.readcol(edge_file,'iif',delimiter='\t')
    col1 = np.array(col1)
    col2 = np.array(col2)
    edge_list = zip(col1, col2)
    
    nodes, clusters = utilities.readcol(community_file, 'ii',delimiter='\t')
    nodes = np.array(nodes)
    community_list = zip(nodes, clusters)

    # Create graph
    LFR_graph = nx.DiGraph(edge_list)

    # Assigne community values
    for node in LFR_graph.nodes():
        LFR_graph.node[node]['community'] = community_list[node-1][1]

    return LFR_graph

def unweighted_directed_lfr_graph(N, mu, k, maxk, minc, maxc, deg_exp=1.0, 
    com_exp=1.0, on=0, om=0, temp_dir_ID=0, full_path=None, benchmark_file=None):
    """
    Creates a temporary directory in which to generate an LFR graph, then removes the directory
    and returns a networkx graph
    """

    if full_path != None:
        path = full_path
        directory = path + "temp_" + str(temp_dir_ID) + "/"
        params={'N':N, 'k':k, 'maxk':maxk, 'mu':mu, 't1':deg_exp, 't2':com_exp, 
            'minc':minc, 'maxc':maxc, 'on':on, 'om':om, 'param_file':directory+'flags.dat'}

        if not os.path.exists(directory):
            os.makedirs(directory)
        if benchmark_file != None:
            command_file = path + benchmark_file
        else:
            command_file = path + "directed_benchmark"
        shutil.copy(command_file, directory)
        os.chdir(directory)

        graph = generate_graph(params, command_file, directory)

    else:
        path = './'
        directory = path + "temp_" + str(temp_dir_ID) + "/"
        params={'N':N, 'k':k, 'maxk':maxk, 'mu':mu, 't1':deg_exp, 't2':com_exp, 
            'minc':minc, 'maxc':maxc, 'on':on, 'om':om, 'param_file':path+'flags.dat'}

        if not os.path.exists(directory):
            os.makedirs(directory)
        if benchmark_file != None:
            command_file = path + benchmark_file
        else:
            command_file = path + "directed_benchmark"
        shutil.copy(command_file, directory)
        os.chdir(directory)

        graph = generate_graph(params, command_file, path)

    # move back
    if full_path:
        os.chdir(full_path)
    else:
        os.chdir('..')
    # del temp dir
    shutil.rmtree(directory)

    return graph

def unweighted_directed_lfr_graph_asarray(**kwargs):

    return np.asarray(nx.to_numpy_matrix(unweighted_directed_lfr_graph(**kwargs)))

if __name__ == '__main__':
    """
    """
    
    nx.write_gexf(make_lfr_graph(5000, 0.1, 5, 5, 20, 20), 'test.gexf')