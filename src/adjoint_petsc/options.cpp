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

#include <iostream>

#include <adjoint_petsc/options.h>

AP_NAMESPACE_START
struct ADPetscOptions {
  bool          debug_abbriviate_identifiers       = false;
  bool          debug_output_primal                = false;
  bool          debug_output_reverse               = true;
  bool          debug_output_identifiers           = true;
  bool          debug_output_obj_info_on_all_ranks = false;
  int           debug_output_precission            = 12;
  std::ostream* debug_output_stream                = &std::cerr;
};

ADPetscOptions ad_gobal_options = {};

bool          ADPetscOptionsGetDebugAbbreviateIdentifiers() { return ad_gobal_options.debug_abbriviate_identifiers; }
bool          ADPetscOptionsGetDebugOutputPrimal() { return ad_gobal_options.debug_output_primal; }
bool          ADPetscOptionsGetDebugOutputReverse() { return ad_gobal_options.debug_output_reverse; }
bool          ADPetscOptionsGetDebugOutputIdentifiers() { return ad_gobal_options.debug_output_identifiers; }
bool          ADPetscOptionsGetDebugOutputObjInfoOnAllRanks() { return ad_gobal_options.debug_output_obj_info_on_all_ranks; }
int           ADPetscOptionsGetDebugOutputPrecission() { return ad_gobal_options.debug_output_precission; }
std::ostream& ADPetscOptionsGetDebugOutputStream() { return *ad_gobal_options.debug_output_stream; }

void ADPetscOptionsSetDebugAbbreviateIdentifiers(bool value) { ad_gobal_options.debug_abbriviate_identifiers = value; }
void ADPetscOptionsSetDebugOutputPrimal(bool value) { ad_gobal_options.debug_output_primal = value; }
void ADPetscOptionsSetDebugOutputReverse(bool value) { ad_gobal_options.debug_output_reverse = value; }
void ADPetscOptionsSetDebugOutputIdentifiers(bool value) { ad_gobal_options.debug_output_identifiers = value; }
void ADPetscOptionsSetDebugOutputObjInfoOnAllRanks(bool value) { ad_gobal_options.debug_output_obj_info_on_all_ranks = value; }
void ADPetscOptionsSetDebugOutputPrecission(int value) { ad_gobal_options.debug_output_precission = value; }
void ADPetscOptionsSetDebugOutputStream(std::ostream& value) { ad_gobal_options.debug_output_stream = &value; }

AP_NAMESPACE_END
