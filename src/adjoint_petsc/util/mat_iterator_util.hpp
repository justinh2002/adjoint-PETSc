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

#pragma once

#include <adjoint_petsc/mat.h>

#include "../impl/ad_mat_data.h"

#include "exception.hpp"

AP_NAMESPACE_START

namespace iterator_implementation {
  template<typename T, typename = void>
  struct MatSeqAIJValueAccess {

    T& value(PetscInt offset);
  };

  template<>
  struct MatSeqAIJValueAccess<Mat> {
    Real* values;

    MatSeqAIJValueAccess(Mat mat) : values(nullptr) {
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat, nullptr, nullptr, &values, nullptr));
    }

    Real& value(PetscInt offset) { return values[offset]; }
  };

  template<>
  struct MatSeqAIJValueAccess<ADMat> {
    MatSeqAIJValueAccess<Mat> mat_v;
    ADMatSeqAIJData*          mat_i;

    std::array<char, sizeof(Wrapper)> wrapper;

    MatSeqAIJValueAccess(ADMat mat) : mat_v(mat->mat), mat_i(ADMatSeqAIJData::cast(mat->mat_i)) {}

    Wrapper& value(PetscInt offset) {
      new (wrapper.data()) Wrapper(mat_v.value(offset), mat_i->index[offset]);

      return *reinterpret_cast<Wrapper*>(wrapper.data());
    }
  };

  template<>
  struct MatSeqAIJValueAccess<ADMatData*> {
    ADMatSeqAIJData*          data;

    MatSeqAIJValueAccess(ADMatData* data) : data(ADMatSeqAIJData::cast(data)) {}

    Identifier& value(PetscInt offset) {
      return data->index[offset];
    }
  };

  template<typename Func, typename ... Values>
  PetscErrorCode iterateMatSeqAIJI(Func&& func, Mat mat, PetscInt const* colmap, MatSeqAIJValueAccess<Values>&& ... values) {

    PetscInt const* row_offset;
    PetscInt const* column_ids;
    PetscMemType mem;
    PetscInt low;
    PetscInt high;
    PetscInt row_range;

    PetscCall(MatGetOwnershipRange(mat, &low, &high));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, nullptr, &mem));

    row_range = high - low;

    for(int cur_local_row = 0; cur_local_row < row_range; cur_local_row += 1) {
      int col_range = row_offset[cur_local_row + 1] - row_offset[cur_local_row];
      int col_offset = row_offset[cur_local_row];

      int row = cur_local_row + low;
      for(int cur_local_col = 0; cur_local_col < col_range; cur_local_col += 1) {
        int col = column_ids[cur_local_col + col_offset];
        if(nullptr != colmap) {
          col = colmap[col];
        } else {
          col += low;
        }
        func(row, col, values.value(cur_local_col + col_offset)...);
      }
    }

    return PETSC_SUCCESS;
  }

  template<typename T, typename = void>
  struct MatAIJValueAccess {

    MatSeqAIJValueAccess<T> accessMatD();
    MatSeqAIJValueAccess<T> accessMatO();
  };

  template<>
  struct MatAIJValueAccess<Mat> {
    Mat mat;

    Mat matd;
    Mat mato;

    MatAIJValueAccess() = default;

    MatAIJValueAccess(Mat mat_) : MatAIJValueAccess() {
      mat = mat_;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, nullptr));
    }

    auto accessMatD() { return MatSeqAIJValueAccess<Mat>(matd); };
    auto accessMatO() { return MatSeqAIJValueAccess<Mat>(mato); };
  };

  template<>
  struct MatAIJValueAccess<ADMat> {
    ADMat mat;

    ADMatImpl matd;
    ADMatImpl mato;

    MatAIJValueAccess() = default;

    MatAIJValueAccess(ADMat mat_) : MatAIJValueAccess() {
      mat = mat_;

      PetscCallVoid(MatMPIAIJGetSeqAIJ(mat->mat, &matd.mat, &mato.mat, nullptr));

      ADMatMPIAIJData* data = ADMatMPIAIJData::cast(mat->mat_i);
      matd.mat_i = &data->index_d;
      mato.mat_i = &data->index_o;
    }

    auto accessMatD() { return MatSeqAIJValueAccess<ADMat>(&matd); };
    auto accessMatO() { return MatSeqAIJValueAccess<ADMat>(&mato); };
  };

  template<>
  struct MatAIJValueAccess<ADMatData*> {
    ADMatData* datad;
    ADMatData* datao;

    MatAIJValueAccess() = default;

    MatAIJValueAccess(ADMatData* d) : MatAIJValueAccess() {
      ADMatMPIAIJData* data = ADMatMPIAIJData::cast(d);
      datad = &data->index_d;
      datao = &data->index_o;
    }

    auto accessMatD() { return MatSeqAIJValueAccess<ADMatData*>(datad); };
    auto accessMatO() { return MatSeqAIJValueAccess<ADMatData*>(datao); };
  };


  template<typename Func, typename ... Values>
  PetscErrorCode iterateMatAIJ(Func&& func, Mat mat, MatAIJValueAccess<Values>&& ... values) {

    Mat matd;
    Mat mato;
    PetscInt const* colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

    PetscInt col_low;
    PetscInt col_high;

    PetscInt row_low;
    PetscInt row_high;
    PetscCall(MatGetOwnershipRange(mat, &row_low, &row_high));
    PetscCall(MatGetOwnershipRangeColumn(mat, &col_low, &col_high));

    auto func_shift = [&] (PetscInt row, PetscInt col, auto&& ... values) {
      func(row + row_low, col + col_low, std::forward<decltype(values)>(values)...);
    };

    auto func_shift_row = [&] (PetscInt row, PetscInt col, auto&& ... values) {
      func(row + row_low, col, std::forward<decltype(values)>(values)...);
    };

    PetscCall(iterateMatSeqAIJI(func_shift,     matd, nullptr, values.accessMatD()...));
    PetscCall(iterateMatSeqAIJI(func_shift_row, mato, colmap, values.accessMatO()...));

    return PETSC_SUCCESS;
  }

  template<typename Func, typename ... Values>
  PetscErrorCode accessMatSeqAIJ(Func&& func, PetscInt row, PetscInt col, Mat mat, PetscInt const* colmap, MatSeqAIJValueAccess<Values>&& ... values) {

    PetscInt const* row_offset;
    PetscInt const* column_ids;
    PetscMemType mem;

    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, nullptr, &mem));

    int cur_local_row = row;
    int col_range = row_offset[cur_local_row + 1] - row_offset[cur_local_row];
    int col_offset = row_offset[cur_local_row];

    for(int cur_local_col = 0; cur_local_col < col_range; cur_local_col += 1) {
      int cur_col = column_ids[cur_local_col + col_offset];
      if(nullptr != colmap) {
        cur_col = colmap[cur_col];
      }

      if(cur_col == col) {
        func(values.value(cur_local_col + col_offset)...);

        return PETSC_SUCCESS;
      }
    }

    return PETSC_ERR_USER_INPUT;
  }

  template<typename Func, typename ... Values>
  PetscErrorCode accessMatAIJ(Func&& func, PetscInt row, PetscInt col, Mat mat, MatAIJValueAccess<Values>&& ... values) {

    Mat matd;
    Mat mato;
    PetscInt const* colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

    PetscInt col_low;
    PetscInt col_high;
    PetscCall(MatGetOwnershipRangeColumn(mat, &col_low, &col_high));
    PetscInt row_low;
    PetscInt row_high;
    PetscCall(MatGetOwnershipRange(mat, &row_low, &row_high));

    if(col_low <= col && col < col_high) {
      PetscCall(accessMatSeqAIJ(std::forward<Func>(func), row - row_low, col - col_low, matd, nullptr, values.accessMatD()...));
    } else {
      auto r = accessMatSeqAIJ(std::forward<Func>(func), row - row_low, col, mato, colmap, values.accessMatO()...);
      PetscCall(r);
    }

    return PETSC_SUCCESS;
  }

  inline Mat getUnderlyingMat(Mat   mat) { return mat; }
  inline Mat getUnderlyingMat(ADMat mat) { return mat->mat; }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode MatIterateAllEntries(Func&& func, First&& mat, Other&& ... other) {
  ADMatType type = ADMatGetADType(mat);

  if(ADMatType::ADMatMPIAIJ == type) {
    return iterator_implementation::iterateMatAIJ(
        func,
        iterator_implementation::getUnderlyingMat(std::forward<First>(mat)),
        iterator_implementation::MatAIJValueAccess<std::remove_cv_t<std::remove_reference_t<First>>>(std::forward<First>(mat)),
        iterator_implementation::MatAIJValueAccess<std::remove_cv_t<std::remove_reference_t<Other>>>(std::forward<Other>(other))...
      );
  }
  else if(ADMatType::ADMatSeqAIJ == type) {
    return iterator_implementation::iterateMatSeqAIJI(
        func,
        iterator_implementation::getUnderlyingMat(std::forward<First>(mat)),
        nullptr,
        iterator_implementation::MatSeqAIJValueAccess<std::remove_cv_t<std::remove_reference_t<First>>>(std::forward<First>(mat)),
        iterator_implementation::MatSeqAIJValueAccess<std::remove_cv_t<std::remove_reference_t<Other>>>(std::forward<Other>(other))...
      );
  }
  else {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode MatAccessValue(Func&& func, PetscInt row, PetscInt col, First&& mat, Other&& ... other) {
  ADMatType type = ADMatGetADType(mat);

  if(ADMatType::ADMatMPIAIJ == type) {
    return iterator_implementation::accessMatAIJ(
        func,
        row,
        col,
        iterator_implementation::getUnderlyingMat(std::forward<First>(mat)),
        iterator_implementation::MatAIJValueAccess<std::remove_cv_t<std::remove_reference_t<First>>>(std::forward<First>(mat)),
        iterator_implementation::MatAIJValueAccess<std::remove_cv_t<std::remove_reference_t<Other>>>(std::forward<Other>(other))...
      );
  }
  else if(ADMatType::ADMatSeqAIJ == type) {
    return iterator_implementation::accessMatSeqAIJ(
        func,
        row,
        col,
        iterator_implementation::getUnderlyingMat(std::forward<First>(mat)),
        nullptr,
        iterator_implementation::MatSeqAIJValueAccess<std::remove_cv_t<std::remove_reference_t<First>>>(std::forward<First>(mat)),
        iterator_implementation::MatSeqAIJValueAccess<std::remove_cv_t<std::remove_reference_t<Other>>>(std::forward<Other>(other))...
      );
  }
  else {
    return PETSC_ERR_ARG_WRONGSTATE;
  }


}

AP_NAMESPACE_END
