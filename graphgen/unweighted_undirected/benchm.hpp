//
// Created by nathaniel on 4/24/18.
//

#ifndef GRAPHGEN_BENCHM_HPP
#define GRAPHGEN_BENCHM_HPP

#include <deque>
#include <set>

void build_network(int num_nodes, double average_k, int max_degree, double tau,
                   double tau2, double mixing_parameter, int overlapping_nodes,
                   int overlap_membership, int nmin, int nmax, bool fixed_range,
                   bool excess, bool defect, bool randomf, int seed,
                   double clustering_coeff,
                   std::deque<std::set<int> > &output_Eout,
                   std::deque<std::deque<int> > &output_member_list);

#endif //GRAPHGEN_BENCHM_HPP
