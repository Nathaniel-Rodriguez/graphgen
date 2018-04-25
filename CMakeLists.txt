cmake_minimum_required(VERSION 3.9)
project(graphgen)

set(CMAKE_CXX_STANDARD 11)

add_executable(graphgen
        graphgen/unweighted_directed/benchm.cpp
        graphgen/unweighted_directed/benchm.hpp
        graphgen/unweighted_directed/cast.cpp
        graphgen/unweighted_directed/combinatorics.cpp
        graphgen/unweighted_directed/histograms.cpp
        graphgen/unweighted_directed/print.cpp
        graphgen/unweighted_directed/random.cpp
        graphgen/unweighted_directed/set_parameters.cpp
        graphgen/unweighted_directed/standard_include.cpp
        graphgen/unweighted_directed/unweighted_directed_graph_generator.cpp
        graphgen/unweighted_directed/unweighted_directed_graph_generator.hpp
        graphgen/unweighted_undirected/benchm.cpp
        graphgen/unweighted_undirected/cast.cpp
        graphgen/unweighted_undirected/cc.cpp
        graphgen/unweighted_undirected/combinatorics.cpp
        graphgen/unweighted_undirected/histograms.cpp
        graphgen/unweighted_undirected/print.cpp
        graphgen/unweighted_undirected/random.cpp
        graphgen/unweighted_undirected/set_parameters.cpp
        graphgen/unweighted_undirected/standard_include.cpp
        graphgen/unweighted_undirected/unweighted_undirected_graph_generator.cpp
        graphgen/weighted_directed/benchm.cpp
        graphgen/weighted_directed/cast.cpp
        graphgen/weighted_directed/combinatorics.cpp
        graphgen/weighted_directed/dir_benchm.cpp
        graphgen/weighted_directed/histograms.cpp
        graphgen/weighted_directed/print.cpp
        graphgen/weighted_directed/random.cpp
        graphgen/weighted_directed/set_parameters.cpp
        graphgen/weighted_directed/standard_include.cpp
        graphgen/weighted_directed/weighted_directed_graph_generator.cpp
        graphgen/weighted_undirected/benchm.cpp
        graphgen/weighted_undirected/binary_benchm.cpp
        graphgen/weighted_undirected/cast.cpp
        graphgen/weighted_undirected/cc.cpp
        graphgen/weighted_undirected/combinatorics.cpp
        graphgen/weighted_undirected/histograms.cpp
        graphgen/weighted_undirected/print.cpp
        graphgen/weighted_undirected/random.cpp
        graphgen/weighted_undirected/set_parameters.cpp
        graphgen/weighted_undirected/standard_include.cpp)
