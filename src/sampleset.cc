// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "sampleset.h"
#include <Eigen/StdVector>

namespace libgp {
  
  SampleSet::SampleSet (int input_dim)
  {
    this->input_dim = input_dim;
    n = 0;
  }
  
  SampleSet::SampleSet ( const SampleSet& ss )
  {
    // shallow copies
    n = ss.n;
    input_dim = ss.input_dim;
    targets = ss.targets;

    // deep copy needed for vector of pointers
    for (size_t i=0; i<ss.inputs.size(); ++i)
    {
      Eigen::VectorXd * sample_to_store = new Eigen::VectorXd(input_dim);
      *sample_to_store = *ss.inputs.at(i);
      inputs.push_back(sample_to_store);
    }
  }

  SampleSet::~SampleSet() 
  {
    clear();
  }
  
  void SampleSet::add(const double x[], double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  
  void SampleSet::add(const Eigen::VectorXd x, double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(x);
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  
  const Eigen::VectorXd & SampleSet::x(size_t k)
  {
      return *inputs.at(k);
  }

  double SampleSet::getInputValue(size_t k, size_t j) const
  {
      Eigen::VectorXd * in = inputs.at(k);
      return (*in)(j);
  }

  double SampleSet::y(size_t k)
  {
    return targets.at(k);
  }

  const std::vector<double>& SampleSet::y() 
  {
    return targets;
  }

  bool SampleSet::set_y(size_t i, double y)
  {
    if (i>=n) return false;
    targets[i] = y;
    return true;
  }
  
  size_t SampleSet::size()
  {
    return n;
  }
  
  void SampleSet::clear()
  {
    while (!inputs.empty()) {
      delete inputs.back();
      inputs.pop_back();
    }    
    n = 0;
    targets.clear();
  }
  
  bool SampleSet::empty ()
  {
      return n==0;
  }

  double SampleSet::minL1DistanceToKnown(const std::vector<double> &x) const
  {
      if (x.size() != input_dim) {
          throw std::runtime_error(std::string("Size of vector (") + std::to_string(x.size()) + ") doesn't match input_dim ("
                                   + std::to_string(input_dim) + ") in method SampleSet::minL1DistanceToKnown.");
      }
      //const Eigen::VectorXd _x(x.data(), input_dim);
      //Eigen::Map<Eigen::VectorXd> _x(x.data(), input_dim);
      //const Eigen::VectorXd _x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x.data(), input_dim);
      Eigen::VectorXd _x(input_dim);
      for (size_t ii = 0; ii < input_dim; ++ii) {
          _x(ii) = x[ii];
      }
      return minL1DistanceToKnown(_x);
  }

  double SampleSet::minL1DistanceToKnown(const Eigen::VectorXd &x) const
  {
      double result = std::numeric_limits<double>::max();
      for (Eigen::VectorXd * const sample : inputs) {
          double const norm = (x - (*sample)).lpNorm<1>();
          if (norm < result) {
              result = norm;
          }
      }
      return result;
  }
}
