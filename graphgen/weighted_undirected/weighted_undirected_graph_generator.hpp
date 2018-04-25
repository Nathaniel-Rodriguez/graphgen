//
// Created by nathaniel on 4/24/18.
//

#ifndef GRAPHGEN_WEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP
#define GRAPHGEN_WEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP

#include <Python.h>
#include <deque>
#include <map>
#include <set>

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list);
PyObject* ConvertWeightMapToNumpyArray(const std::deque<std::map<int, double> >& Wout,
                                       const std::deque<std::set<int> >& edge_list);
PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& edge_list);
PyMODINIT_FUNC PyInit_weighted_undirected_graph_generator(void);

#endif //GRAPHGEN_WEIGHTED_UNDIRECTED_GRAPH_GENERATOR_HPP
