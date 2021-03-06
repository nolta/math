#ifdef STAN_OPENCL
#include <stan/math/prim/mat.hpp>
#include <stan/math/gpu/copy.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixGPU, zero_m_exception_pass) {
  stan::math::matrix_gpu m(1, 1);

  EXPECT_NO_THROW(m.zeros<stan::math::TriangularViewGPU::Entire>());
  EXPECT_NO_THROW(m.zeros<stan::math::TriangularViewGPU::Lower>());
  EXPECT_NO_THROW(m.zeros<stan::math::TriangularViewGPU::Upper>());

  stan::math::matrix_gpu m0;
  EXPECT_NO_THROW(m0.zeros<stan::math::TriangularViewGPU::Entire>());
  EXPECT_NO_THROW(m0.zeros<stan::math::TriangularViewGPU::Lower>());
  EXPECT_NO_THROW(m0.zeros<stan::math::TriangularViewGPU::Upper>());
}

TEST(MathMatrixGPU, zero_m_value_check) {
  stan::math::matrix_d m0(2, 2);
  stan::math::matrix_d m0_dst(2, 2);
  m0 << 2, 2, 2, 2;
  stan::math::matrix_gpu m(m0);
  stan::math::matrix_gpu m_upper(m0);
  stan::math::matrix_gpu m_lower(m0);

  EXPECT_NO_THROW(m.zeros<stan::math::TriangularViewGPU::Entire>());
  EXPECT_NO_THROW(m_lower.zeros<stan::math::TriangularViewGPU::Lower>());
  EXPECT_NO_THROW(m_upper.zeros<stan::math::TriangularViewGPU::Upper>());

  stan::math::copy(m0_dst, m);
  EXPECT_EQ(0, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(0, m0_dst(1, 1));

  stan::math::copy(m0_dst, m_lower);
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(2, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));

  stan::math::copy(m0_dst, m_upper);
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(2, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));
}

#endif
