// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __RPROP_H__
#define __RPROP_H__

#include "gp.h"
#include <Eigen/Core>
#include <random>

namespace libgp {

/** Gradient-based optimizer.
 *  @author Manuel Blum */
class RProp
{
public:
  RProp () {init();}
  void init(double eps_stop = 0.0, double Delta0=0.1, double Deltamin=1e-6, double Deltamax=50, double etaminus=0.5, double etaplus=1.2, double min_stepsize_factor = 1e-1);
  void maximize(GaussianProcess * gp, size_t n=100, bool verbose=true, bool print_params = false);
  bool optimize(GaussianProcess * gp, size_t n=100, bool verbose=true, bool print_params = false);
  void minimize_crossvalidation(GaussianProcess * gp, size_t n=100, bool verbose=true, bool print_params = false);
private:
  std::mt19937_64 generator;
  double Delta0;
  double Deltamin;
  double Deltamax;
  double etaminus;
  double etaplus;
  double eps_stop;
  double min_stepsize_factor;

  bool is_finite(const Eigen::VectorXd& x);
  bool make_feasible(GaussianProcess * gp, const size_t max_it = 100);
  bool make_crossvalidation_feasible(GaussianProcess * gp, const size_t max_it = 100);
};
}

#endif /* __RPROP_H__ */
