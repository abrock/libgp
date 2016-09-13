// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <random>

#include "rprop.h"
#include "gp_utils.h"


namespace libgp {

void RProp::init(double eps_stop, double Delta0, double Deltamin, double Deltamax, double etaminus, double etaplus, double min_stepsize_factor)
{
  this->Delta0   = Delta0;
  this->Deltamin = Deltamin;
  this->Deltamax = Deltamax;
  this->etaminus = etaminus;
  this->etaplus  = etaplus;
  this->eps_stop = eps_stop;
  this->min_stepsize_factor = min_stepsize_factor;
}

bool RProp::isfinite(const Eigen::VectorXd& x) {
  for (int ii = 0; ii < x.size(); ++ii) {
    if (!std::isfinite(x(ii))) {
      return false;
    }
  }
  return true;
}

bool RProp::make_feasible(GaussianProcess * gp, const size_t max_it){

    const Eigen::VectorXd orig_params = gp->covf().get_loghyper();

    const int param_dim = gp->covf().get_param_dim();
    std::vector<double> test_x (param_dim);

    double scale = 1.0;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(-1,1);

    for (size_t ii = 0; ii < max_it; ++ii) {
        Eigen::VectorXd current_params(orig_params);
        for (int jj = 0; jj < param_dim; ++jj) {
            current_params(jj) += scale * uniform(generator);
        }
        gp->covf().set_loghyper(current_params);
        if (std::isfinite(gp->f(test_x.data()))) {
            //std::cout << "Made process feasible in iteration " << ii << std::endl;
            return true;
        }

        for (int jj = 0; jj < param_dim; ++jj) {
            current_params(jj) = scale * uniform(generator);
        }
        gp->covf().set_loghyper(current_params);
        if (std::isfinite(gp->f(test_x.data()))) {
            //std::cout << "Made process feasible in iteration " << ii << std::endl;
            return true;
        }
    }

    gp->covf().set_loghyper(orig_params);
    std::cout << "Failed to make process feasible wrt. one evaluation" << std::endl;
    std::cout << "Current parameters: [" << orig_params;
    for (int ii = 1; ii < orig_params.size(); ++ii) {
        std::cout << ", " << orig_params(ii);
    }
    std::cout << "]" << std::endl;
    return false;
}

bool RProp::make_crossvalidation_feasible(GaussianProcess * gp, const size_t max_it){

    const Eigen::VectorXd orig_params = gp->covf().get_loghyper();

    const int param_dim = static_cast<int>(gp->covf().get_param_dim());

    double scale_low = 1.0;
    double scale_high = 1.0;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(-1,1);

    if (std::isfinite(gp->cross_validation())) {
        return true;
    }

    std::vector<double> scales(2, 1.0);

    for (size_t ii = 0; ii < max_it; ++ii) {
        scales[0] = scale_low;
        scales[1] = scale_high;
        for (const double scale : scales) {
            Eigen::VectorXd current_params(orig_params);
            for (int jj = 0; jj < param_dim; ++jj) {
                current_params(jj) += scale * uniform(generator);
            }
            gp->covf().set_loghyper(current_params);

            if (std::isfinite(gp->cross_validation())) {
                //std::cout << "Made process feasible in iteration " << ii << std::endl;
                return true;
            }

            for (int jj = 0; jj < param_dim; ++jj) {
                current_params(jj) = scale * uniform(generator);
            }
            gp->covf().set_loghyper(current_params);

            if (std::isfinite(gp->cross_validation())) {
                //std::cout << "Made process feasible in iteration " << ii << std::endl;
                return true;
            }

            for (int jj = 0; jj < param_dim; ++jj) {
                current_params(jj) = std::log(scale)/std::log(10);
            }
            gp->covf().set_loghyper(current_params);

            if (std::isfinite(gp->cross_validation())) {
                //std::cout << "Made process feasible in iteration " << ii << std::endl;
                return true;
            }
        }

        scale_low  *= 1.01;
        scale_high *= 0.99;
    }

    gp->covf().set_loghyper(orig_params);
    /*
    std::cout << "Failed to make process feasible wrt. crossvalidation" << std::endl;
    std::cout << "Current parameters: [" << orig_params(0);
    for (int ii = 1; ii < orig_params.size(); ++ii) {
        std::cout << ", " << orig_params(ii);
    }
    std::cout << "]" << std::endl;
    */
    return false;
}

void RProp::maximize(GaussianProcess * gp, size_t n, bool verbose, bool print_params)
{
  const int param_dim = gp->covf().get_param_dim();
  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
  Eigen::VectorXd params = gp->covf().get_loghyper();
  Eigen::VectorXd best_params = params;
  double best = gp->log_likelihood();

  float stepsize_factor = 1.0;


  for (size_t i=0; i<n; ++i) {
    Eigen::VectorXd grad = -stepsize_factor * gp->log_likelihood_gradient();
    if (!isfinite(grad)) {
        make_feasible(gp);
        params = best_params = gp->covf().get_loghyper();
        best = gp->log_likelihood();
        continue;
    }
    grad_old = grad_old.cwiseProduct(grad);
    for (int j=0; j<grad_old.size(); ++j) {
      if (grad_old(j) > 0) {
        Delta(j) = std::min(Delta(j)*etaplus, Deltamax);
      } else if (grad_old(j) < 0) {
        Delta(j) = std::max(Delta(j)*etaminus, Deltamin);
        grad(j) = 0;
      }
      params(j) += -Utils::sign(grad(j)) * Delta(j);
    }
    grad_old = grad;
    if (grad_old.norm() < eps_stop) break;
    gp->covf().set_loghyper(params);
    const double lik = gp->log_likelihood();
    if (verbose) std::cout << i << " " << -lik << std::endl;
    if (print_params) {
      std::cout << "[" << params[0];
      for (int jj = 1; jj < params.size(); ++jj) {
        std::cout << ", " << params[jj];
      }
      std::cout << "]" << std::endl;
    }
    if (lik > best && std::isfinite(lik)) {
      best = lik;
      best_params = params;
    }
    else {
      stepsize_factor /= 2;
      if (verbose) std::cout << "no improvement in step " << i
                             << ", reduced stepsize_factor to " << stepsize_factor << std::endl;
      if (stepsize_factor < min_stepsize_factor) {
        if (verbose) std::cout << "stepsize_factor below threshold, finishing" << std::endl;
        break;
      }
    }
  }
  gp->covf().set_loghyper(best_params);
}

void RProp::minimize_crossvalidation(GaussianProcess * gp, size_t n, bool verbose, bool print_params)
{
  const int param_dim = gp->covf().get_param_dim();
  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
  Eigen::VectorXd params = gp->covf().get_loghyper();
  Eigen::VectorXd best_params = params;
  Eigen::VectorXd best_crossvalidation_params = params;

  double best_crossvalidation = gp->cross_validation();

  for (double entry = -4; entry < 4; entry += .5) {
      Eigen::VectorXd current_params = Eigen::VectorXd::Zero(param_dim);
      for (int ii = 0; ii < param_dim; ++ii) {
          current_params(ii) = entry;
      }
      gp->covf().set_loghyper(current_params);
      const double current_crossvalidation = gp->cross_validation();
      if (!std::isfinite(best_crossvalidation) && std::isfinite(current_crossvalidation)) {
          best_crossvalidation = current_crossvalidation;
          best_crossvalidation_params = current_params;
          best_params = current_params;
      }
      if (current_crossvalidation < best_crossvalidation && std::isfinite(current_crossvalidation)) {
        best_crossvalidation = current_crossvalidation;
        best_crossvalidation_params = current_params;
        best_params = current_params;
      }
  }
  double best = gp->log_likelihood();

  float stepsize_factor = 1.0;


  for (size_t i=0; i<n; ++i) {
    Eigen::VectorXd grad = -stepsize_factor * gp->log_likelihood_gradient();
    if (!isfinite(grad)) {
        if (verbose) {
            std::cout << "### Process not feasible, adjusting...";
        }
        make_feasible(gp);
        if (verbose) {
          std::cout << " done. Old parameters: " << std::endl;
          std::cout << "[" << params[0];
          for (int jj = 1; jj < params.size(); ++jj) {
            std::cout << ", " << params[jj];
          }
          std::cout << "]" << std::endl;
        }
        params = best_params = gp->covf().get_loghyper();
        const double current_crossvalidation = gp->cross_validation();
        if (current_crossvalidation < best_crossvalidation && std::isfinite(current_crossvalidation)) {
            best_crossvalidation = current_crossvalidation;
            best_crossvalidation_params = params;
        }
        best = gp->log_likelihood();
        if (verbose) {
          std::cout << "New parameters:" << std::endl;
          std::cout << "[" << params[0];
          for (int jj = 1; jj < params.size(); ++jj) {
            std::cout << ", " << params[jj];
          }
          std::cout << "]" << std::endl;
        }
        continue;
    }
    grad_old = grad_old.cwiseProduct(grad);
    for (int j=0; j<grad_old.size(); ++j) {
      if (grad_old(j) > 0) {
        Delta(j) = std::min(Delta(j)*etaplus, Deltamax);
      } else if (grad_old(j) < 0) {
        Delta(j) = std::max(Delta(j)*etaminus, Deltamin);
        grad(j) = 0;
      }
      params(j) += -Utils::sign(grad(j)) * Delta(j);
    }
    grad_old = grad;
    if (grad_old.norm() < eps_stop) break;
    gp->covf().set_loghyper(params);
    const double lik = gp->log_likelihood();
    const double crossvalidation = gp->cross_validation();
    if (verbose) std::cout << i << " loglikelihood: " << -lik
                           << ", cross error: " << crossvalidation << std::endl;

    if (!std::isfinite(crossvalidation)) {
        stepsize_factor /= 2;
        if (stepsize_factor < min_stepsize_factor) {
          if (verbose) std::cout << "stepsize_factor below threshold, finishing" << std::endl;
          break;
        }
        if (verbose) {
            std::cout << "### Process not feasible, adjusting...";
        }
        make_crossvalidation_feasible(gp);
        if (verbose) {
          std::cout << " done. Old parameters: " << std::endl;
          std::cout << "[" << params[0];
          for (int jj = 1; jj < params.size(); ++jj) {
            std::cout << ", " << params[jj];
          }
          std::cout << "]" << std::endl;
        }
        params = best_params = gp->covf().get_loghyper();
        const double current_crossvalidation = gp->cross_validation();
        if (current_crossvalidation < best_crossvalidation && std::isfinite(current_crossvalidation)) {
            best_crossvalidation = current_crossvalidation;
            best_crossvalidation_params = params;
        }
        best = gp->log_likelihood();
        if (verbose) {
          std::cout << "New parameters:" << std::endl;
          std::cout << "[" << params[0];
          for (int jj = 1; jj < params.size(); ++jj) {
            std::cout << ", " << params[jj];
          }
          std::cout << "]" << std::endl;
        }
        continue;
    }
    if (print_params) {
      std::cout << "[" << params[0];
      for (int jj = 1; jj < params.size(); ++jj) {
        std::cout << ", " << params[jj];
      }
      std::cout << "]" << std::endl;
    }
    if (crossvalidation < best_crossvalidation && std::isfinite(crossvalidation)) {
        best_crossvalidation = crossvalidation;
        best_crossvalidation_params = params;
    }
    if (lik > best && std::isfinite(lik)) {
      best = lik;
      best_params = params;
    }
    else {
      stepsize_factor /= 2;
      if (verbose) std::cout << "no improvement in step " << i
                             << ", reduced stepsize_factor to " << stepsize_factor << std::endl;
      if (stepsize_factor < min_stepsize_factor) {
        if (verbose) std::cout << "stepsize_factor below threshold, finishing" << std::endl;
        break;
      }
    }
  }
  gp->covf().set_loghyper(best_crossvalidation_params);
  if (!std::isfinite(gp->cross_validation())) {
      std::cout << "Crossvalidation not finite after optimization" << std::endl;
  }
}

} // namespace libgp
