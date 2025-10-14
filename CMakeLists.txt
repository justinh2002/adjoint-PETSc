#
# adjoint-PETSc
#
# Copyright (C) 2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
# Homepage: http://scicomp.rptu.de
# Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
#
# Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#
# This file is part of adjoint-PETSc (GITHUB_LINK).
#
# adjoint-PETSc is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# adjoint-PETSc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU
# Lesser General Public License along with adjoint-PETSc.
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#

cmake_minimum_required(VERSION 3.20)
project(adjoint_petsc
  VERSION 0.1.0
  LANGUAGES CXX)

include(CTest)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CoDiPack_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/codipack/cmake CACHE PATH "Path to CoDiPack cmake files.")
set(PETSc_DIR     Not_Defined CACHE PATH "Path to petsc")

option(BUILD_EXAMPLES    "Build examples"               ON)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

list(APPEND CMAKE_PREFIX_PATH "${PETSc_DIR}")

find_package(CoDiPack CONFIG REQUIRED)
find_package(MPI REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET GLOBAL PETSc)

add_library(adjoint_petsc
  src/adjoint_petsc/ksp.cpp
  src/adjoint_petsc/mat.cpp
  src/adjoint_petsc/options.cpp
  src/adjoint_petsc/util/addata_helper.cpp
  src/adjoint_petsc/util/petsc_missing.cpp
  src/adjoint_petsc/vec.cpp
  src/adjoint_petsc/impl/ad_mat_data.cpp
  src/adjoint_petsc/impl/ad_vec_data.cpp
)

target_sources(adjoint_petsc
  PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS include
    FILES
      include/adjoint_petsc/ksp.h
      include/adjoint_petsc/mat.h
      include/adjoint_petsc/options.h
      include/adjoint_petsc/vec.h
      include/adjoint_petsc/util/base.hpp
      include/adjoint_petsc/util/codi_def.hpp
      include/adjoint_petsc/util/wrapper_array.hpp
)

target_include_directories(adjoint_petsc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options    (adjoint_petsc PRIVATE -Wall -pedantic -Wextra -Werror)

target_link_libraries(adjoint_petsc PUBLIC CoDiPack)
target_link_libraries(adjoint_petsc PUBLIC PkgConfig::PETSc MPI::MPI_CXX)

install(TARGETS adjoint_petsc FILE_SET api)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
