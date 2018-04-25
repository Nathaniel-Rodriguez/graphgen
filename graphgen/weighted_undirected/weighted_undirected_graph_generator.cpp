/*
 * Provides access in Python to LFR graphs generated from c++ code
 */

#include <Python.h>
#include <cstddef>
#include <deque>
#include <set>
#include <map>
#include <iostream>
#include <vector>
#include "numpy/arrayobject.h"
#include "benchm.hpp"
#include "weighted_undirected_graph_generator.hpp"

/*
 * Calls LFR c++ code and passes arguments to it so it can build the network.
 * The network is returned as a numpy edge array and a tuple w/ community
 * assignments for each node.
 */
static PyObject *GenerateWeightedUndirectedGraph(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"num_nodes", "average_k", "max_degree",
                                 "mixing_parameter", "mixing_parameter2",
                                 "com_size_min",
                                 "com_size_max", "seed", "tau", "tau2",
                                 "overlapping_nodes", "overlap_membership",
                                 "fixed_range", "excess", "defect",
                                 "randomf", "beta", "clustering_coeff", NULL};

  int num_nodes;
  double average_k;
  int max_degree;
  double mixing_parameter;
  double mixing_parameter2;
  int nmin;
  int nmax;
  int seed;
  double beta(1.5);
  double clustering_coeff(0.0);
  double tau(2.0);
  double tau2(1.0);
  int overlapping_nodes(0);
  int overlap_membership(0);
  int fixed_range(0); // bool
  int excess(0); // bool
  int defect(0); // bool
  int randomf(0); // bool

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ididdiii|ddiiiiiidd", 
                                   keyword_list,
                                   &num_nodes, &average_k, &max_degree,
                                   &mixing_parameter, &mixing_parameter2,
                                   &nmin, &nmax, &seed,
                                   &tau, &tau2, &overlapping_nodes,
                                   &overlap_membership,
                                   &fixed_range, &excess, &defect, &randomf,
                                   &beta, &clustering_coeff)) {
    std::cerr << "Error parsing GenerateWeightedUndirectedGraph arguments" << std::endl;
    return NULL;
  }

  std::deque<std::set<int> > edge_list;
  std::deque<std::deque<int> > member_list;
  std::deque<std::map<int, double> > weights;
  build_network(num_nodes, average_k, max_degree, tau, tau2, mixing_parameter,
                mixing_parameter2, beta,
                overlapping_nodes, overlap_membership, nmin, nmax,
                static_cast<bool>(fixed_range), static_cast<bool>(excess),
                static_cast<bool>(defect), static_cast<bool>(randomf),
                seed, clustering_coeff, edge_list, member_list, weights);

  PyObject* edge_array = ConvertEdgeDequeToNumpyArray(edge_list);
  PyObject* weight_array = ConvertWeightMapToNumpyArray(weights, edge_list);
  PyObject* member_tuple = ConvertMemberDequeToTuple(member_list);
  PyObject* return_tuple = PyTuple_New(3);
  PyTuple_SetItem(return_tuple, 0, edge_array);
  PyTuple_SetItem(return_tuple, 1, member_tuple);
  PyTuple_SetItem(return_tuple, 2, weight_array);

  return return_tuple;
}

PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& edge_list) {
  // Determine the number of edges for the array
  npy_intp num_edges(0);
  for (auto head_set = edge_list.begin(); head_set != edge_list.end(); ++head_set) {
    for (auto head = head_set->begin(); head != head_set->end(); ++head) {
      ++num_edges;
    }
  }

  // Generate dimensional information for Ex2 edge array
  std::vector<npy_intp> size_data(2);
  size_data[0] = num_edges;
  size_data[1] = 2;

  // Create new numpy array
  PyObject* py_array = PyArray_SimpleNew(2, size_data.data(), NPY_UINT64);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_uint64* data = reinterpret_cast<npy_uint64*>(np_array->data);

  // Fill numpy array with edges
  std::size_t edge_index(0);
  for (std::size_t tail = 0; tail < edge_list.size(); ++tail) {
    for (auto head = edge_list[tail].begin(); head != edge_list[tail].end(); ++head) {
      data[edge_index] = tail;
      data[edge_index+1] = *head;
      edge_index += 2;
    }
  }

  return py_array;
}

PyObject* ConvertWeightMapToNumpyArray(const std::deque<std::map<int, double> >& weights,
                                       const std::deque<std::set<int> >& edge_list) {
  // Determine the number of edges for the array
  npy_intp num_edges(0);
  for (auto head_set = edge_list.begin(); head_set != edge_list.end(); ++head_set) {
    for (auto head = head_set->begin(); head != head_set->end(); ++head) {
      ++num_edges;
    }
  }

  // Generate dimensional information for weight array
  std::vector<npy_intp> size_data(1);
  size_data[0] = num_edges;

  // Create new numpy array
  PyObject* py_array = PyArray_SimpleNew(1, size_data.data(), NPY_FLOAT32);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_float32* data = reinterpret_cast<npy_float32*>(np_array->data);

  // Fill numpy array with edges
  std::size_t edge_index(0);
  for (std::size_t tail = 0; tail < edge_list.size(); ++tail) {
    for (auto head = edge_list[tail].begin(); head != edge_list[tail].end(); ++head) {
      data[edge_index] = static_cast<npy_float32>(weights[tail].at(*head));
      edge_index += 1;
    }
  }

  return py_array;
}

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list) {

  PyObject* member_tuple = PyTuple_New(static_cast<Py_ssize_t>(member_list.size()));

  // For each node loop through and make a tuple of its members
  for (Py_ssize_t member = 0; member < PyTuple_Size(member_tuple); ++member) {
    PyObject* communities_tuple = PyTuple_New(static_cast<Py_ssize_t>(
                                              member_list[member].size()));

    // Add community memberships to nodes tuple
    for (Py_ssize_t iii = 0; iii < PyTuple_Size(communities_tuple); ++iii) {
      PyTuple_SetItem(communities_tuple, iii, PyLong_FromLong(
      static_cast<long>(member_list[member][iii])));
    }

    PyTuple_SetItem(member_tuple, member, communities_tuple);
  }

  return member_tuple;
}

static PyMethodDef WeightedUndirectedGraphGeneratorMethods[] = {
{ "GenerateWeightedUndirectedGraph", (PyCFunction) GenerateWeightedUndirectedGraph,
              METH_VARARGS | METH_KEYWORDS,
"Creates weighted undirected LFR graphs. Returns Ex2 numpy edge "
"array and tuple of community assignments"},
{ NULL, NULL, 0, NULL}
};

static struct PyModuleDef WeightedUndirectedGraphGeneratorModule = {
PyModuleDef_HEAD_INIT,
"weighted_undirected_graph_generator",
"Creates weighted undirected LFR graphs",
-1,
WeightedUndirectedGraphGeneratorMethods
};

PyMODINIT_FUNC PyInit_weighted_undirected_graph_generator(void) {
  import_array();
  return PyModule_Create(&WeightedUndirectedGraphGeneratorModule);
}