//
// Created by nathaniel on 4/24/18.
//

#ifndef GRAPHGEN_BENCHM_HPP
#define GRAPHGEN_BENCHM_HPP

#include <deque>
#include <set>
#include <map>

void build_network(int num_nodes, double average_k, int max_degree, double tau,
                   double tau2, double mixing_parameter,
                   double mixing_parameter2, double beta, int overlapping_nodes,
                   int overlap_membership, int nmin, int nmax, bool fixed_range,
                   bool excess, bool defect, bool randomf, int seed,
                   std::deque<std::set<int> > &output_Eout,
                   std::deque<std::deque<int> > &output_member_list,
                   std::deque<std::map <int, double > > &output_Wout);

#endif //GRAPHGEN_BENCHM_HPP
