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

#include <adjoint_petsc/vec.h>

#include "../impl/ad_vec_data.h"

AP_NAMESPACE_START

namespace iterator_implementation {
  template<typename T, typename = void>
  struct VecLocalValueAccess {

    T& value(PetscInt offset);
  };

  template<>
  struct VecLocalValueAccess<Vec> {
    Vec vec;
    Real* values;

    VecLocalValueAccess(Vec vec) : vec(vec), values(nullptr) {
      PetscCallVoid(VecGetArray(vec, &values));
    }

    ~VecLocalValueAccess() {
      PetscCallVoid(VecRestoreArray(vec, &values));
    }

    Real& value(PetscInt offset) { return values[offset]; }
  };

  template<>
  struct VecLocalValueAccess<ADVec> {
    ADVec vec;
    WrapperArray values;

    std::array<char, sizeof(Wrapper)> wrapper;

    VecLocalValueAccess(ADVec vec) : vec(vec), values() {
      PetscCallVoid(VecGetArray(vec, &values));
    }

    ~VecLocalValueAccess() {
      PetscCallVoid(VecRestoreArray(vec, &values));
    }

    Wrapper& value(PetscInt offset) {
      // TODO: Improve wrapper.
      Wrapper temp = values[offset];
      new (wrapper.data()) Wrapper(temp.value(), temp.getIdentifier());

      return *reinterpret_cast<Wrapper*>(wrapper.data());
    }
  };

  template<>
  struct VecLocalValueAccess<ADVecData*> {
    ADVecData* data;
    Identifier* ids;

    VecLocalValueAccess(ADVecData* data) : data(data), ids(data->getArray()) {}

    ~VecLocalValueAccess() {
      data->restoreArray(ids);
    }

    Identifier& value(PetscInt offset) { return ids[offset]; }
  };

  template<typename Func, typename ... Values>
  PetscErrorCode iterateVecAll(Func&& func, Vec vec, VecLocalValueAccess<Values>&& ... values) {

    PetscInt low;
    PetscInt high;
    PetscInt range;

    PetscCall(VecGetOwnershipRange(vec, &low, &high));
    range = high - low;

    for(PetscInt i = 0; i < range; i += 1) {
      func(i + low, values.value(i)...);
    }
    return PETSC_SUCCESS;
  }

  template<typename Func, typename ... Values>
  PetscErrorCode iterateVecIndexSet(Func&& func, PetscInt n, PetscInt const* ix, Vec vec, VecLocalValueAccess<Values>&& ... values) {

    PetscInt low;
    PetscCall(VecGetOwnershipRange(vec, &low, nullptr));

    for(PetscInt i = 0; i < n; i += 1) {
      func(i, ix[i], values.value(ix[i] - low)...);
    }
    return PETSC_SUCCESS;
  }

  inline Vec getUnderlyingVec(Vec   vec) { return vec; }
  inline Vec getUnderlyingVec(ADVec vec) { return vec->vec; }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode VecIterateAllEntries(Func&& func, First&& vec, Other&& ... other) {
  ADVecType type = ADVecGetADType(vec);

  if(ADVecType::ADVecMPI == type || ADVecType::ADVecSeq == type) {
    return iterator_implementation::iterateVecAll(
        func,
        iterator_implementation::getUnderlyingVec(std::forward<First>(vec)),
        iterator_implementation::VecLocalValueAccess<std::remove_reference_t<First>>(std::forward<First>(vec)),
        iterator_implementation::VecLocalValueAccess<std::remove_reference_t<Other>>(std::forward<Other>(other))...
      );
  }
  else {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode VecIterateIndexSet(Func&& func, PetscInt n, PetscInt const* ix, First&& vec, Other&& ... other) {
  ADVecType type = ADVecGetADType(vec);

  if(ADVecType::ADVecMPI == type || ADVecType::ADVecSeq == type) {
    return iterator_implementation::iterateVecIndexSet(
        func,
        n,
        ix,
        iterator_implementation::getUnderlyingVec(std::forward<First>(vec)),
        iterator_implementation::VecLocalValueAccess<std::remove_cv_t<std::remove_reference_t<First>>>(std::forward<First>(vec)),
        iterator_implementation::VecLocalValueAccess<std::remove_cv_t<std::remove_reference_t<Other>>>(std::forward<Other>(other))...
        );
  }
  else {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
}

AP_NAMESPACE_END
