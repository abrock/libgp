// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 *  
 *   \page licence Licensing
 *    
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum */
  class GaussianProcess
  {
  public:

    /** Empty initialization */
    GaussianProcess ();
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename);

    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const std::string filename);

    /** Copy constructor */
    GaussianProcess (const GaussianProcess& gp);
    
    virtual ~GaussianProcess ();
    
    /** Write current gp model to file. */
    void write(const char * filename);

    /** Write current gp model to file. */
    void write(const std::string filename);
    
    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double f(const double x[]);
    
    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double var(const double x[]);

    /**
     * Predict both target value and variance for given input
     * @param[in] x input vector
     * @param[out] var predicted variance
     * @return predicted function value
     */
    double eval(const double x[], double& var);
    
    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    bool add_pattern(const double x[], double y);

    double cross_validation(std::vector<double>& errors, double& variance);
    double cross_validation(double& variance);
    double cross_validation(std::vector<double>& errors);
    double cross_validation();

    bool set_y(size_t i, double y);

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();
    
    /** Clear sample set and free memory. */
    void clear_sampleset();
    
    /** Get reference on currently used covariance function. */
    CovarianceFunction & covf();
    
    /** Get input vector dimensionality. */
    size_t get_input_dim();

    double log_likelihood();
    
    Eigen::VectorXd log_likelihood_gradient();

  protected:
    
    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    
    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
//    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::MatrixXd L;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute();
    
    bool alpha_needs_update;

    bool reject_duplicates = true;

  private:

    /** No assignement */
    GaussianProcess& operator=(const GaussianProcess&);

  };
}

#endif /* __GP_H__ */
