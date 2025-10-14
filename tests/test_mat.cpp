/*
 * adjoint-PETSc
 *
 * Copyright (C) 2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://scicomp.rptu.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of adjoint-PETSc (GITHUB_LINK).
 *
 * adjoint-PETSc is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * adjoint-PETSc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License for more details.
 * You should have received a copy of the GNU
 * Lesser General Public License along with adjoint-PETSc.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 */

#include <gtest/gtest.h>

#include <adjoint_petsc/util/mat_iterator_util.hpp>

#include "setups/mat_setup.hpp"


TEST_F(MatSetup, SetValues) {
  std::array<PetscInt, 4> rows = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank * ENTRIES_PER_RANK + 2,      mpi_rank_next * ENTRIES_PER_RANK + 3};
  std::array<PetscInt, 4> cols = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank_next * ENTRIES_PER_RANK + 2, mpi_rank * ENTRIES_PER_RANK + 3};
  std::array<adjoint_petsc::Number, 4> values = {s[0], s[1], s[2], s[3]};

  for(int i = 0;i < 4; i += 1) {
    PetscCallVoid(MatSetValues(mat[0], 1, &rows[i], 1, &cols[i], &values[i], ADD_VALUES));
  }

  PetscCallVoid(MatAssemblyBegin(mat[0], MAT_FINAL_ASSEMBLY));
  PetscCallVoid(MatAssemblyEnd(mat[0], MAT_FINAL_ASSEMBLY));


  auto func = [&](PetscInt row, PetscInt col, adjoint_petsc::Wrapper& value) {
    tape->registerOutput(value);

    if(0 == row && col == 0) {
      EXPECT_EQ(value.getValue(), 12.0);
      value.setGradient(100 + 10 * mpi_rank);
    } else if(1 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank * 10 + 2);
      value.setGradient(1000 + 10 * mpi_rank);
    } else if(2 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank * 10 + 3);
      value.setGradient(10000 + 10 * mpi_rank);
    } else if(3 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank_prev * 10 + 4);
      value.setGradient(100000 + 10 * mpi_rank);
    } else {
      AP_EXCEPTION("Wrong row/col access.");
    }
  };
  adjoint_petsc::MatIterateAllEntries(func, mat[0]);

  evaluateTape();

  EXPECT_EQ(s[0].getGradient(), 100.0); // Adjoint from first rank.
  EXPECT_EQ(s[1].getGradient(), 1000.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[2].getGradient(), 10000.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[3].getGradient(), 100000.0 + 10 * mpi_rank_next); // Adjoint from next rank.
}

TEST_F(MatSetup, SetValuesInsert) {
  std::array<PetscInt, 3> rows = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank * ENTRIES_PER_RANK + 1};
  std::array<PetscInt, 3> cols = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank * ENTRIES_PER_RANK + 1};
  std::array<adjoint_petsc::Number, 3> values = {s[0], s[1], s[2]};

  int start = mpi_rank == 0 ? 0 : 1; // Only set first value from first rank
  for(int i = start;i < 3; i += 1) {
    PetscCallVoid(MatSetValues(mat[0], 1, &rows[i], 1, &cols[i], &values[i], INSERT_VALUES));
  }

  PetscCallVoid(MatAssemblyBegin(mat[0], MAT_FINAL_ASSEMBLY));
  PetscCallVoid(MatAssemblyEnd(mat[0], MAT_FINAL_ASSEMBLY));


  auto func = [&](PetscInt row, PetscInt col, adjoint_petsc::Wrapper& value) {
    tape->registerOutput(value);

    if(0 == row && col == 0) {
      EXPECT_EQ(value.getValue(), 1.0);
      value.setGradient(100 + 10 * mpi_rank);
    } else if(1 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank * 10 + 3);
      value.setGradient(1000 + 10 * mpi_rank);
    } else {
      AP_EXCEPTION("Wrong row/col access.");
    }
  };
  adjoint_petsc::MatIterateAllEntries(func, mat[0]);

  evaluateTape();

  EXPECT_EQ(s[0].getGradient(), mpi_rank == 0 ? 100.0 : 0.0); // Adjoint from first rank. Distributed to all, since the setting of the value is a race condition.
  EXPECT_EQ(s[1].getGradient(), 0); // Overwritten
  EXPECT_EQ(s[2].getGradient(), 1000.0 + 10 * mpi_rank); // Adjoint from own rank.
}

TEST_F(MatSetup, Duplicate) {

  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> right;
  diag.fill(s[0]);
  left.fill(s[1]);
  right.fill(s[2]);
  PetscCallVoid(initTriDiagMatrix(mat[0], diag.data(), left.data(), right.data()));

  adjoint_petsc::ADMat c;
  PetscCallVoid(MatDuplicate(mat[0], MAT_COPY_VALUES, &c));

  auto func = [&](PetscInt row, PetscInt col, adjoint_petsc::Wrapper& value) {
    tape->registerOutput(value);

    if( row == col) {
      EXPECT_EQ(value.getValue(), 1.0 + 10 * mpi_rank);
      value.setGradient(100 + 10 * mpi_rank);
    } else if((row + 1) % (ENTRIES_PER_RANK * mpi_size) == col) {
      EXPECT_EQ(value.getValue(), 3.0 + 10 * mpi_rank);
      value.setGradient(1000 + 10 * mpi_rank);
    } else {
      if(0 == row % ENTRIES_PER_RANK) {
        EXPECT_EQ(value.getValue(), 2.0 + 10 * mpi_rank_prev);
      } else {
        EXPECT_EQ(value.getValue(), 2.0 + 10 * mpi_rank);
      }
      value.setGradient(10000 + 10 * mpi_rank);
    }
  };
  adjoint_petsc::MatIterateAllEntries(func, c);

  evaluateTape();

  int sum = (mpi_size - 1) * mpi_size / 2; // 0 + 1 + ... + (mpi_size - 1)
  EXPECT_EQ(s[0].getGradient(), ENTRIES_PER_RANK * (100.0 + 10.0 * mpi_rank));
  EXPECT_EQ(s[1].getGradient(), (ENTRIES_PER_RANK - 1) * (10000.0 + 10.0 * mpi_rank) + (10000.0 + 10.0 * mpi_rank_next));
  EXPECT_EQ(s[2].getGradient(), ENTRIES_PER_RANK * (1000.0 + 10.0 * mpi_rank));

  PetscCallVoid(MatDestroy(&c));
}

TEST_F(MatSetup, GetValues) {

  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> right;
  diag.fill(s[0]);
  left.fill(s[1]);
  right.fill(s[2]);
  PetscCallVoid(initTriDiagMatrix(mat[0], diag.data(), left.data(), right.data()));

  std::array<PetscInt, 4> indices = {ENTRIES_PER_RANK * mpi_rank, ENTRIES_PER_RANK * mpi_rank + 1, ENTRIES_PER_RANK * mpi_rank + 2, ENTRIES_PER_RANK * mpi_rank + 3};
  std::array<adjoint_petsc::Number, 8> values;

  PetscCallVoid(MatGetValues(mat[0], 2, &indices[0], 2, &indices[0], &values[0])); // First block
  PetscCallVoid(MatGetValues(mat[0], 2, &indices[2], 2, &indices[2], &values[4])); // Second block

  adjoint_petsc::Number temp = 42;
  PetscCallVoid(MatGetValue(mat[0], -1, 0, &temp));
  EXPECT_EQ(temp.getValue(), 42); // Value is not modified.
  PetscCallVoid(MatGetValue(mat[0], 0, -1, &temp));
  EXPECT_EQ(temp.getValue(), 42); // Value is not modified.


  std::array<adjoint_petsc::Real, 4> expected_values = {s[0].getValue(), s[2].getValue(), s[1].getValue(), s[0].getValue()};
  for(int i = 0; i < 4; i += 1) {
    EXPECT_EQ(values[i].getValue(), expected_values[i]);
    EXPECT_EQ(values[i + 4].getValue(), expected_values[i]);

    tape->registerOutput(values[i]);
    tape->registerOutput(values[i + 4]);

    values[i].setGradient(100 + 10 * mpi_rank);
    values[i + 4].setGradient(1000 + 10 * mpi_rank);
  }

  evaluateTape();

  EXPECT_EQ(s[0].getGradient(), 2 * 100 + 2 * 1000 + 4 * (10 * mpi_rank));
  EXPECT_EQ(s[1].getGradient(), 100 + 1000 + 2 * (10 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), 100 + 1000 + 2 * (10 * mpi_rank));
}

TEST_F(MatSetup, Mult) {

  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> right;
  diag.fill(s[0]);
  left.fill(s[1]);
  right.fill(s[2]);
  PetscCallVoid(initTriDiagMatrix(mat[0], diag.data(), left.data(), right.data()));

  PetscCallVoid(VecSet(vec[0], s[3]));

  PetscCallVoid(MatMult(mat[0], vec[0], vec[1]));

  std::array<adjoint_petsc::Real, ENTRIES_PER_RANK * 2> expected_values = {184, 24, 24, 54, 344, 504, 504, 374};
  if(2 != mpi_size) {
    AP_EXCEPTION("Test only implemented for two mpi ranks.");
  }

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[1], &values));

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    adjoint_petsc::Wrapper temp = values[i];
    EXPECT_EQ(temp.getValue(), expected_values[i + mpi_rank * ENTRIES_PER_RANK]);

    tape->registerOutput(temp);

    temp.setGradient(100 + 10 * mpi_rank);
  }

  PetscCallVoid(VecRestoreArray(vec[1], &values));

  evaluateTape();

  std::array<adjoint_petsc::Real, ENTRIES_PER_RANK * 2> expected_gradients = {1730, 600, 600, 620, 2830, 3960, 3960, 3840};

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(values[i].getValue(), expected_values[i + mpi_rank * ENTRIES_PER_RANK]);

    values[i].setGradient(100 + 10 * mpi_rank);
  }

  std::array<adjoint_petsc::Real, 2> expected_gradients_s0 = {1600, 6160};
  std::array<adjoint_petsc::Real, 2> expected_gradients_s1 = {1640, 6020};
  std::array<adjoint_petsc::Real, 2> expected_gradients_s2 = {2600, 5060};
  std::array<adjoint_petsc::Real, 2> expected_gradients_s3 = {3550, 14590};
  EXPECT_EQ(s[0].getGradient(), expected_gradients_s0[mpi_rank]);
  EXPECT_EQ(s[1].getGradient(), expected_gradients_s1[mpi_rank]);
  EXPECT_EQ(s[2].getGradient(), expected_gradients_s2[mpi_rank]);
  EXPECT_EQ(s[3].getGradient(), expected_gradients_s3[mpi_rank]);
}

template<typename Test>
void performNormTest(Test& test, NormType type, adjoint_petsc::Real expected_value, adjoint_petsc::Real* expected_grad) {
  std::array<adjoint_petsc::Number, Test::ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, Test::ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, Test::ENTRIES_PER_RANK> right;
  diag.fill(test.s[0]);
  left.fill(test.s[1]);
  right.fill(-test.s[2]);
  PetscCallVoid(test.initTriDiagMatrix(test.mat[0], diag.data(), left.data(), right.data()));

  adjoint_petsc::Number norm;
  PetscCallVoid(MatNorm(test.mat[0], type, &norm));

  EXPECT_DOUBLE_EQ(norm.getValue(), expected_value);
  test.tape->registerOutput(norm);
  norm.setGradient(100 + 10 * test.mpi_rank);

  test.evaluateTape();

  EXPECT_DOUBLE_EQ(test.s[0].getGradient(), expected_grad[0]);
  EXPECT_DOUBLE_EQ(test.s[1].getGradient(), expected_grad[1]);
  EXPECT_DOUBLE_EQ(test.s[2].getGradient(), expected_grad[2]);
}

TEST_F(MatSetup, Norm_1) {
  adjoint_petsc::Real expected_value = 36;
  std::array<adjoint_petsc::Real, 3> expected_gradients_zero = {0.0, 0.0, 0.0};

  int mpi_sum = 10 * (mpi_size - 1) * mpi_size / 2;
  adjoint_petsc::Real grad_base = (100.0 * mpi_size + mpi_sum) * (ENTRIES_PER_RANK - 1);
  std::array<adjoint_petsc::Real, 3> expected_gradients = {grad_base, grad_base, grad_base};
  if (0 == mpi_rank_next) {
    performNormTest(*this, NORM_1, expected_value, expected_gradients.data());
  }
  else {
    performNormTest(*this, NORM_1, expected_value, expected_gradients_zero.data());
  }
}

TEST_F(MatSetup, Norm_F) {
  if(2 != mpi_size) {
    AP_EXCEPTION("Test only implemented for two mpi ranks.");
  }

  adjoint_petsc::Real expected_value = 42.33202097703346;
  std::array<adjoint_petsc::Real, 3 * 2> expected_gradients = {
      19.84313483298442, 39.68626966596885, 59.52940449895328,
      218.2744831628287, 238.1176179958131, 257.96075282879758
  };
  performNormTest(*this, NORM_FROBENIUS, expected_value, &expected_gradients[mpi_rank * 3]);

}

TEST_F(MatSetup, Norm_INF) {
  adjoint_petsc::Real expected_value = 36;
  std::array<adjoint_petsc::Real, 3> expected_gradients_zero = {0.0, 0.0, 0.0};

  int mpi_sum = 10 * (mpi_size - 1) * mpi_size / 2;
  adjoint_petsc::Real grad_base = (100.0 * mpi_size + mpi_sum) * (ENTRIES_PER_RANK - 1);
  std::array<adjoint_petsc::Real, 3> expected_gradients = {grad_base, grad_base, grad_base};
  if (0 == mpi_rank_next) {
    performNormTest(*this, NORM_INFINITY, expected_value, expected_gradients.data());
  }
  else {
    performNormTest(*this, NORM_INFINITY, expected_value, expected_gradients_zero.data());
  }
}

TEST_F(MatSetup, ZeroEntries) {
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> right;
  diag.fill(s[0]);
  left.fill(s[1]);
  right.fill(s[2]);
  PetscCallVoid(initTriDiagMatrix(mat[0], diag.data(), left.data(), right.data()));

  PetscCallVoid(MatZeroEntries(mat[0]));

  auto func = [&] (PetscInt row, PetscInt col, adjoint_petsc::Wrapper& value) {
    tape->registerOutput(value);

    EXPECT_EQ(value.getIdentifier(), 0);
  };
  MatIterateAllEntries(func, mat[0]);

  evaluateTape();

  EXPECT_EQ(s[0].getGradient(), 0.0);
  EXPECT_EQ(s[1].getGradient(), 0.0);
  EXPECT_EQ(s[2].getGradient(), 0.0);
}
