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

  struct Result {
    double f;
    double var;
    double expected_improvement;
    Result(double const _f = 0, double const _var = 1, double const _ex = 0) : f(_f), var(_var), expected_improvement(_ex) {}
  };
  
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

    const SampleSet& get_sample_set() const;

    size_t get_call_counter() {return call_counter;}
    
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

    /**
     * Predict both target value and variance for given input
     * @param[in] x input vector
     * @param[out] var predicted variance
     * @return predicted function value
     */
    double eval(const std::vector<double>& x, double& var);

    double minL1DistanceToKnown(const std::vector<double>& x);

    /**
     * @brief Calculate the expected improvement of a target function modelled by the GP
     * @param prediction Prediction of the gp (f())
     * @param variance Variance of the GP (var())
     * @param best_known Best known value of the target function (lower is better)
     * @return
     */
    double expectedImprovement(const double prediction,
                               const double variance,
                               const double best_known);

    /**
     * @brief Calculate the expected improvement of a target function modelled by the GP
     * @param x Parameter vector.
     * @return
     */
    double expectedImprovement(const std::vector<double>& x);

    /**
     * @brief Calculate the expected improvement of a target function modelled by the GP
     * @param x Parameter vector.
     * @return
     */
    double expectedImprovement(const double x[]);

    /**
     * @brief getResult calculates prediction, variance and expected improvement and returns all three as object of type "Result"
     * @param x
     * @return
     */
    Result getResult(const std::vector<double>& x);

    /**
     * @brief getResult calculates prediction, variance and expected improvement and returns all three as object of type "Result"
     * @param x
     * @return
     */
    Result getResult(const double x[]);

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

    /**
     * @brief set_scales Sets the factors by which each of the input parameters is scaled before processing.
     * @param new_scales Vector of factors
     */
    void set_scales(const std::vector<double>& new_scales);
    void set_scales(const Eigen::VectorXd& new_scales);

    /**
     * @brief set_target_center Sets the center value for the target values.
     * The center value is subtracted from the y values in the training samples before training
     * and added to the result given by the Gaussian process before returning it.
     * @param new_center New center value.
     */
    void set_target_center(const double new_center);

  protected:

    bool use_scaling = false;

    /**
     * @brief scales Scales for preprocessing the data.
     */
    Eigen::VectorXd scales;

    /**
     * @brief y_center Center value of the target, is removed
     */
    double y_center = 0;
    
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

    /**
     * @brief call_counter Number of times eval(), f() or var() has been called since initialisation.
     */
    size_t call_counter = 0;

    double lowest_y;

  private:

    /** No assignement */
    GaussianProcess& operator=(const GaussianProcess&);

  };
}

#endif /* __GP_H__ */
