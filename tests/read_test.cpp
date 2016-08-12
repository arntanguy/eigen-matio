#define BOOST_TEST_MODULE ReadTest

#include <boost/test/unit_test.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <complex>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <MATio/MATio.hpp>
#include <map>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

template<typename TensorType>
bool isApprox(const TensorType& t1, const TensorType& t2, const double epsilon = 10e-6)
{
  Tensor<double, 0> res = (t1-t2).square().abs().sum();
  return res(0) < 10e-6;
}

struct ReadFixture
{
  const double epsilon = 10e-6;
};

struct ReadRealFixture : public ReadFixture
{
  std::map<std::string, Eigen::MatrixXd> mat;
  const std::string file = "test.mat";

  ReadRealFixture()
  {
    mat["r14"].resize(1, 4);
    mat["r14"] << 0.859442, 0.805489, 0.576722, 0.182922;

    mat["r41"].resize(4, 1);
    mat["r41"] << 0.479922, 0.904722, 0.609867, 0.617666;

    mat["r42"].resize(4, 2);
    mat["r42"] << 0.903721, 0.19781, 0.890923, 0.0305409, 0.334163, 0.744074, 0.698746, 0.500022;
  }
};

struct ReadComplexFixture : public ReadFixture
{
  std::map<std::string, Matrix<std::complex<double>, Dynamic, Dynamic>> mat;
  const double epsilon = 10e-6;
  const std::string file = "test.mat";

  ReadComplexFixture()
  {
    mat["c14"].resize(1, 4);
    mat["c14"].real() << 0.99715, 0.972018, 0.750809, 0.855213;
    mat["c14"].imag() << -0.0754396, 0.234904, -0.66052, -0.518277;

    mat["c41"].resize(4, 1);
    mat["c41"].real() << 0.0632169, 0.756359, 0.983814, -0.997988;
    mat["c41"].imag() << 0.998, -0.654156, 0.179192, 0.0634089;

    mat["c42"].resize(4, 2);
    mat["c42"].real() << -0.427325, -0.568524, 0.65396, -0.99933, -0.1798, 0.181525, 0.783322, -0.217916;
    mat["c42"].imag() << -0.904098, -0.822667, 0.756529, 0.0365981, -0.983703, -0.983386, 0.621616, -0.975968;
  }
  // c42
};

struct ReadTensorFixture : public ReadFixture
{
  std::map<std::string, Eigen::Tensor<double, 3>> mat;
  const std::string file = "tensor.mat";

  ReadTensorFixture()
  {
    Eigen::Tensor<double, 3> t(2, 2, 2);
    t.setValues({{{1.,1.},{2.,2.}}, {{3.,3.},{4.,4.}}});
    std::cout << "Tensor " << t << std::endl;
    mat["A"] = t;
  };
};



BOOST_AUTO_TEST_CASE(test_read_real)
{
  std::cout << "Running read test on real matrices" << std::endl;
  ReadRealFixture fix;

  // Open test.mat
  Eigen::MatioFile file(fix.file.c_str(), MAT_ACC_RDONLY, false);

  for (const auto& mat : fix.mat)
  {
    Eigen::MatrixXd matrix;
    file.read_mat(mat.first.c_str(), matrix);
    BOOST_REQUIRE_MESSAGE(matrix.size() == mat.second.size(), "matrix size doesn't match!");
    BOOST_REQUIRE_MESSAGE(matrix.isApprox(mat.second, fix.epsilon), "\nRead:\n" << matrix << "\nExpected:\n"
                                                                                << mat.second);
  }
}

BOOST_AUTO_TEST_CASE(test_read_complex)
{
  std::cout << "Running read test on complex matrices" << std::endl;
  ReadComplexFixture fix;

  // Open test.mat
  Eigen::MatioFile file(fix.file.c_str(), MAT_ACC_RDONLY, false);

  // Read and print the complex matrices
  for (const auto& mat : fix.mat)
  {
    Matrix<std::complex<double>, Dynamic, Dynamic> matrix;
    file.read_mat(mat.first.c_str(), matrix);
    BOOST_REQUIRE_MESSAGE(matrix.size() == mat.second.size(), "matrix size doesn't match!");
    BOOST_REQUIRE_MESSAGE(matrix.isApprox(mat.second, fix.epsilon), "\nRead:\n" << matrix << "\nExpected:\n"
                                                                                << mat.second);
  }
}

BOOST_AUTO_TEST_CASE(test_read_tensor)
{
  ReadTensorFixture fix;

  // Open test.mat
  Eigen::MatioFile file(fix.file.c_str(), MAT_ACC_RDONLY, false);
  for (const auto& mat : fix.mat)
  {
    Tensor<double, 3> tensor;
    file.read_mat(mat.first.c_str(), tensor);
    std::cout << "Read tensor " << tensor << std::endl;
    BOOST_REQUIRE_MESSAGE(tensor.size() == mat.second.size(), "matrix size doesn't match!");
    // TODO write isApprox for tensors
    BOOST_REQUIRE_MESSAGE(isApprox(tensor, mat.second, fix.epsilon), "\nRead:\n" << tensor << "\nExpected:\n"
                                                                                << mat.second);
  }
}
