// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;

  GaussianProcess::GaussianProcess ()
  {
      sampleset = NULL;
      cf = NULL;
  }

  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset = new SampleSet(input_dim);
    L.resize(initial_L_size, initial_L_size);
  }
  
  GaussianProcess::GaussianProcess (const char * filename) 
  {
    sampleset = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
  }

  GaussianProcess::GaussianProcess (const std::string filename) : GaussianProcess(filename.c_str()) {
  }
  
  GaussianProcess::GaussianProcess(const GaussianProcess& gp)
  {
    this->input_dim = gp.input_dim;
    sampleset = new SampleSet(*(gp.sampleset));
    alpha = gp.alpha;
    k_star = gp.k_star;
    alpha_needs_update = gp.alpha_needs_update;
    L = gp.L;
    
    // copy covariance function
    CovFactory factory;
    cf = factory.create(gp.input_dim, gp.cf->to_string());
    cf->loghyper_changed = gp.cf->loghyper_changed;
    cf->set_loghyper(gp.cf->get_loghyper());
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    if (sampleset != NULL) delete sampleset;
    if (cf != NULL) delete cf;
  }  
  
  double GaussianProcess::f(const double _x[])
  {
    if (sampleset->empty()) return 0;
    std::vector<double> x(_x, _x+input_dim);
    if (use_scaling) {
        for (size_t ii = 0; ii < input_dim; ++ii) {
            x[ii] = _x[ii] / scales[static_cast<int>(ii)];
        }
    }
    Eigen::Map<const Eigen::VectorXd> x_star(x.data(), input_dim);

    compute();
    update_alpha();
    update_k_star(x_star);
    return k_star.dot(alpha) + y_center;
  }
  
  double GaussianProcess::var(const double _x[])
  {
    if (sampleset->empty()) return 0;
    std::vector<double> x(_x, _x+input_dim);
    if (use_scaling) {
        for (size_t ii = 0; ii < input_dim; ++ii) {
            x[ii] = _x[ii] / scales[static_cast<int>(ii)];
        }
    }
    Eigen::Map<const Eigen::VectorXd> x_star(x.data(), input_dim);

    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  double GaussianProcess::eval(const double _x[], double& var)
  {
    if (sampleset->empty()) return 0;
    std::vector<double> x(_x, _x+input_dim);
    if (use_scaling) {
        for (size_t ii = 0; ii < input_dim; ++ii) {
            x[ii] = _x[ii] / scales[static_cast<int>(ii)];
        }
    }
    Eigen::Map<const Eigen::VectorXd> x_star(x.data(), input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
    var = cf->get(x_star, x_star) - v.dot(v);
    return k_star.dot(alpha) + y_center;
  }

  void GaussianProcess::compute()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n = sampleset->size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L(i, j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }
  
  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(sampleset->size());
    for(size_t i = 0; i < sampleset->size(); ++i) {
      k_star(i) = cf->get(x_star, sampleset->x(i));
    }
  }

  void GaussianProcess::update_alpha()
  {
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n = sampleset->size();
    alpha = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);
  }
  
  bool GaussianProcess::add_pattern(const double _x[], double y)
  {
    //std::cout<< L.rows() << std::endl;
      y -= y_center;
      std::vector<double> scaled_x(input_dim);
      if (use_scaling) {
          for (size_t ii = 0; ii < input_dim; ++ii) {
              scaled_x[ii] = _x[ii] / scales[static_cast<int>(ii)];
          }
      }
      const double* x = scaled_x.data();
#if 0
    sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    cached_x_star = NULL;
    return;
#else
    int n = static_cast<int>(sampleset->size());
    if (n > 0 && reject_duplicates) {
        // Reject sample if we already have seen it
        Eigen::VectorXd vec_x = Eigen::VectorXd::Zero(static_cast<int>(input_dim));
        for (size_t ii = 0; ii < input_dim; ++ii) {
            vec_x(static_cast<int>(ii)) = x[ii];
        }
        const double self_covariance = cf->get(vec_x, vec_x);
        for (size_t ii = 0; ii < static_cast<size_t>(n); ++ii) {
            if (cf->get(vec_x, sampleset->x(ii)) >= self_covariance) {
                return false;
            }
        }
    }
    sampleset->add(x, y);
    // create kernel matrix if sampleset is empty
    if (n == 0) {
      L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));
      cf->loghyper_changed = false;
    // recompute kernel matrix if necessary
    } else if (cf->loghyper_changed) {
      compute();
    // update kernel matrix 
    } else {
      Eigen::VectorXd k(n);
      for (int i = 0; i<n; ++i) {
        k(i) = cf->get(sampleset->x(i), sampleset->x(n));
      }
      double kappa = cf->get(sampleset->x(n), sampleset->x(n));
      // resize L if necessary
      if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
        L.conservativeResize(n + initial_L_size, n + initial_L_size);
      }
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      L.block(n,0,1,n) = k.transpose();
      L(n,n) = sqrt(kappa - k.dot(k));
    }
    alpha_needs_update = true;
    return true;
#endif
  }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset->set_y(i,y - y_center)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset->clear();
  }
  
  void GaussianProcess::write(const char * filename)
  {
    // output
    std::ofstream outfile;
    outfile.open(filename);
    time_t curtime = time(0);
    tm now=*localtime(&curtime);
    char dest[BUFSIZ]= {0};
    strftime(dest, sizeof(dest)-1, "%c", &now);
    outfile << "# " << dest << std::endl << std::endl
    << "# input dimensionality" << std::endl << input_dim << std::endl 
    << std::endl << "# covariance function" << std::endl 
    << cf->to_string() << std::endl << std::endl
    << "# log-hyperparameter" << std::endl;
    Eigen::VectorXd param = cf->get_loghyper();
    for (size_t i = 0; i< cf->get_param_dim(); i++) {
      outfile << std::setprecision(10) << param(i) << " ";
    }
    outfile << std::endl << std::endl 
    << "# data (target value in first column)" << std::endl;
    for (size_t i=0; i<sampleset->size(); ++i) {
      outfile << std::setprecision(10) << sampleset->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }

  void GaussianProcess::write(const std::string filename) {
    write(filename.c_str());
  }
  
  CovarianceFunction & GaussianProcess::covf()
  {
    return *cf;
  }
  
  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  double GaussianProcess::log_likelihood()
  {
    compute();
    update_alpha();
    int n = static_cast<int>(sampleset->size());
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    const double det = 2 * L.diagonal().head(n).array().log().sum();
    if (!std::isfinite(det)) {
        return -std::numeric_limits<double>::max();
    }
    const double result = -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
    if (!std::isfinite(result)) {
        return -std::numeric_limits<double>::max();
    }
    return result;
  }

  double GaussianProcess::cross_validation(std::vector<double>& errors, double& variance) {
    const size_t size = sampleset->size();
    if (size < 2) {
        return 0;
    }
    errors.clear();
    errors.reserve(size);
    double error_sum = 0;
    for (size_t leftout = 0; leftout < size; ++leftout) {
      const Eigen::VectorXd test_point = sampleset->x(leftout);
      const double test_val = sampleset->y(leftout);
      libgp::GaussianProcess process(get_input_dim(), covf().to_string());
      process.reject_duplicates = false;
      if (use_scaling) {
          process.set_scales(scales);
      }
      process.set_target_center(y_center);
      process.covf().set_loghyper(covf().get_loghyper());
      if (use_scaling) {

      }
      for (size_t ii = 0; ii < size; ++ii) {
        if (leftout != ii) {
          process.add_pattern(sampleset->x(ii).data(), sampleset->y(ii));
        }
      }
      const double prediction = process.f(test_point.data());
      if (!std::isfinite(prediction)) {
          variance = std::numeric_limits<double>::infinity();
          return std::numeric_limits<double>::infinity();
      }
      const double error = std::abs(prediction - test_val);
      error_sum += error;
      errors.push_back(error);
    }
    double square_sum = 0;
    const double mean_error = error_sum / size;
    for (const double error : errors) {
      square_sum += (error - mean_error) * (error - mean_error);
    }
    variance = square_sum / (size - 1);
    return mean_error;
  }

  double GaussianProcess::cross_validation(double& variance) {
    std::vector<double> errors;
    return cross_validation(errors, variance);
  }

  double GaussianProcess::cross_validation(std::vector<double>& errors){
    double variance = 0;
    return cross_validation(errors, variance);
  }

  double GaussianProcess::cross_validation(){
    std::vector<double> errors;
    double variance = 0;
    return cross_validation(errors, variance);
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    update_alpha();
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }


  void GaussianProcess::set_scales(const std::vector<double>& new_scales) {
      if (get_sampleset_size() > 0) {
          throw std::runtime_error("Scales must not be changes after process has been trained");
      }
      const int input_dim = static_cast<int>(covf().get_input_dim());
      scales = Eigen::VectorXd::Zero(input_dim);
      if (static_cast<size_t>(input_dim) > new_scales.size()) {
          throw std::runtime_error("Number of entries in new_scales vector is too low");
      }
      for (int ii = 0; ii < input_dim; ++ii) {
          scales[ii] = new_scales[static_cast<size_t>(ii)];
          if (!std::isfinite(scales[ii])) {
              throw std::runtime_error("New scale is not finite");
          }
          if (0 >= scales[ii]) {
              throw std::runtime_error("New scale is smaller or equal zero");
          }
      }
      use_scaling = true;
  }

  void GaussianProcess::set_scales(const Eigen::VectorXd& new_scales) {
      if (get_sampleset_size() > 0) {
          throw std::runtime_error("Scales must not be changes after process has been trained");
      }
      const int input_dim = static_cast<int>(covf().get_input_dim());
      scales = Eigen::VectorXd::Zero(input_dim);
      if (input_dim > new_scales.size()) {
          throw std::runtime_error("Number of entries in new_scales vector is too low");
      }
      for (int ii = 0; ii < input_dim; ++ii) {
          scales[ii] = new_scales[ii];
          if (!std::isfinite(scales[ii])) {
              throw std::runtime_error("New scale is not finite");
          }
          if (0 >= scales[ii]) {
              throw std::runtime_error("New scale is smaller or equal zero");
          }
      }
      use_scaling = true;
  }

  void GaussianProcess::set_target_center(const double new_center) {
      if (get_sampleset_size() > 0) {
          throw std::runtime_error("Center value must not be changes after process has been trained");
      }
      y_center = new_center;
  }
}



