//
// Created by nathaniel on 4/24/18.
//

#ifndef GRAPHGEN_UNWEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP
#define GRAPHGEN_UNWEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP

#include <Python.h>
#include <deque>

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list);
PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& edge_list);
PyMODINIT_FUNC PyInit_unweighted_undirected_graph_generator(void);

#endif //GRAPHGEN_UNWEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP
