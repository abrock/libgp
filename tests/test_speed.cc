#include "gp.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

TEST(SpeedTest, GP) {

    libgp::GaussianProcess gp(2, "CovSum (CovSEard, CovSEiso)");
    Eigen::VectorXd params(gp.covf().get_param_dim());
    for (int ii = 0; ii < params.size(); ++ii) {
        params[ii] = -.5;
    }
    // set parameters of covariance function
    gp.covf().set_loghyper(params);

    const size_t num_train = 100;
    const size_t num_test = 1e5;

    for (size_t ii = 0; ii < num_train; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        const double y = std::abs(x[0]) + std::abs(x[1]);
        gp.add_pattern(x, y);
    }

    // Sum up everything to prevent the compiler from optimizing important stuff out.
    double trash_sum = 0;

    std::chrono::high_resolution_clock::time_point start, stop;
    start = std::chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        trash_sum += x[0];
        trash_sum += x[1];
    }
    stop = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "creating random arguments took " << time << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        double f = gp.f(x);
        trash_sum += f;
    }
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "evaluating GP function took " << time << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        double var = gp.var(x);
        trash_sum += var;
    }
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "evaluating GP variance took " << time << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        double f = gp.f(x);
        double var = gp.var(x);
        trash_sum += f + var;
    }
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "evaluating GP function + variance via f() and var() took " << time << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        double var = 0;
        double f = gp.eval(x, var);
        trash_sum += f + var;
    }
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "evaluating GP function + variance via eval() took " << time << "s" << std::endl;
}


TEST(SpeedTest, validate) {

    libgp::GaussianProcess gp(2, "CovSum (CovSEard, CovSEiso)");
    Eigen::VectorXd params(gp.covf().get_param_dim());
    for (int ii = 0; ii < params.size(); ++ii) {
        params[ii] = -.5;
    }
    // set parameters of covariance function
    gp.covf().set_loghyper(params);

    const size_t num_train = 100;
    const size_t num_test = 1e5;

    for (size_t ii = 0; ii < num_train; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        const double y = std::abs(x[0]) + std::abs(x[1]);
        gp.add_pattern(x, y);
    }

    for (size_t ii = 0; ii < num_test; ++ii) {
        const double x[] = {drand48()*4-2, drand48()*4-2};
        const double f_old = gp.f(x);
        const double var_old = gp.var(x);
        double var_new = 0;
        const double f_new = gp.eval(x, var_new);
        EXPECT_NEAR(f_old, f_new, 1e-16);
        EXPECT_NEAR(var_old, var_new, 1e-16);
    }
}


