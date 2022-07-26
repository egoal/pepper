#pragma once

#include <Eigen/Core>

namespace water {

struct OptResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::VectorXd x_;
  double obj_;
  int ret_;

  bool IsOptimal() const { return ret_ == 0; }

  std::string ToString() const;
};

///
///@brief solve:
/// min <c, x>
/// s.t. Ax <= b
/// d1 <= x <= d2
///
static OptResult LinearProgramming(const Eigen::VectorXd &c,
                                   const Eigen::MatrixXd &A,
                                   const Eigen::VectorXd &b,
                                   const Eigen::VectorXd &d1,
                                   const Eigen::VectorXd &d2);

} // namespace water