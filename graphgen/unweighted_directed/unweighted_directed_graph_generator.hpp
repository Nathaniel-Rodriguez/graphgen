#ifndef UNWEIGHTED_DIRECTED_GRAPH_GENERATOR_H_
#define UNWEIGHTED_DIRECTED_GRAPH_GENERATOR_H_

#include <Python.h>
#include <deque>

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list);
PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& Eout);
PyMODINIT_FUNC PyInit_unweighted_directed_graph_generator(void);

#endif /* UNWEIGHTED_DIRECTED_GRAPH_GENERATOR_H_ */