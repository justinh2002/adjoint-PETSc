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

#include <adjoint_petsc/mat.h>
#include <adjoint_petsc/options.h>
#include <adjoint_petsc/util/petsc_missing.h>

#include "util/addata_helper.hpp"
#include "util/debug_output.hpp"
#include "util/exception.hpp"
#include "util/mat_iterator_util.hpp"
#include "util/vec_iterator_util.hpp"
#include "util/dyadic_product_helper.hpp"

AP_NAMESPACE_START

/*-------------------------------------------------------------------------------------------------
 * PETSc functions
 */

struct LhsInData {
  PetscInt row;
  PetscInt col;
};

bool operator <(LhsInData const& a, LhsInData const& b) {
  return a.row < b.row || (a.row == b.row && a.col < b.col);
}

bool operator ==(LhsInData const& a, LhsInData const& b) {
  return a.row == b.row && a.col == b.col;
}

struct CommunicationData {
  MPI_Comm comm;
  std::vector<int> request_number_for_rank_send;
  std::vector<int> request_number_for_rank_recv;

  int total_request_send;
  int total_request_recv;

  std::vector<int> send_displs;
  std::vector<int> recv_displs;

  bool active;

  CommunicationData() : comm(MPI_COMM_SELF), request_number_for_rank_send(0), request_number_for_rank_recv(0),
      total_request_send(0), total_request_recv(0), send_displs(0), recv_displs(0), active(false) {}

  void initialize(Mat mat, std::vector<LhsInData> const& lhs_in_positions) {
    PetscInt const* ownership_ranges;
    PetscObjectGetComm((PetscObject)mat, &comm);
    PetscCallVoid(MatGetOwnershipRanges(mat, &ownership_ranges));

    int active_count = (int)lhs_in_positions.size();
    MPI_Allreduce(MPI_IN_PLACE, &active_count, 1, MPI_INTEGER, MPI_SUM, comm);
    active = 0 != active_count;

    if(active) {
      // Step 1: Get number of requested values per rank
      int mpi_rank;
      int mpi_size;
      MPI_Comm_rank(comm, &mpi_rank);
      MPI_Comm_size(comm, &mpi_size);

      // The vector with the lhs_in positions is sorted by rows.
      request_number_for_rank_send.resize(mpi_size);
      int cur_rank = 0;
      for(LhsInData const& cur : lhs_in_positions) {
        // Adjust rank
        while(cur.row >= ownership_ranges[cur_rank + 1]) {
          cur_rank += 1;
        }

        // Add entries
        request_number_for_rank_send[cur_rank] += 1;
      }

      request_number_for_rank_recv.resize(mpi_size);
      MPI_Alltoall(request_number_for_rank_send.data(), 1, MPI_INTEGER, request_number_for_rank_recv.data(), 1, MPI_INTEGER, comm);

      // Step 2: Communicate requested values to ranks
      // step 2.1: Compute the total size and allocate the send/recv displs
      total_request_send = 0;
      total_request_recv = 0;
      send_displs.resize(mpi_size + 1); // One extra for total displacement.
      recv_displs.resize(mpi_size + 1); // One extra for total displacement.
      for(int i = 0; i < mpi_size; i += 1) {
        send_displs[i] = total_request_send;
        recv_displs[i] = total_request_recv;
        total_request_send += request_number_for_rank_send[i];
        total_request_recv += request_number_for_rank_recv[i];
      }
      send_displs[mpi_size] = total_request_send;
      recv_displs[mpi_size] = total_request_recv;
    }
  }

  void commRowCol(LhsInData* send, LhsInData* recv) {
    MPI_Datatype position_type;
    MPI_Type_contiguous(2, MPI_INTEGER, &position_type);
    MPI_Type_commit(&position_type);

    MPI_Alltoallv(send, request_number_for_rank_send.data(), send_displs.data(), position_type,
                  recv, request_number_for_rank_recv.data(), recv_displs.data(), position_type,
                  comm);

    MPI_Type_free(&position_type);
  }

  void commReverse(Real* adjoints_recv, Real* adjoints_send, MPI_Datatype adjoint_type) {
    MPI_Alltoallv(adjoints_recv, request_number_for_rank_recv.data(), recv_displs.data(), adjoint_type,
                  adjoints_send, request_number_for_rank_send.data(), send_displs.data(), adjoint_type,
                  comm);
  }
};

struct ADData_MatSetValues {
  struct RhsData {
    PetscInt lhs_pos;
    Identifier rhs_identifier;
  };

  std::vector<Identifier> lhs_identifiers;
  std::vector<int> lhs_out_positions;

  std::vector<LhsInData> lhs_in_positions;
  std::vector<RhsData> rhs_data;

  std::vector<int> step_boundaries;

  InsertMode mode;

  std::vector<Real> adjoint_data;
  int adjoint_step;

  CommunicationData comm_data;

  ADData_MatSetValues(InsertMode m) : lhs_identifiers(0), lhs_out_positions(0), lhs_in_positions(0),
      rhs_data(0), step_boundaries(0), mode(m), adjoint_data(0), adjoint_step(), comm_data() {
  }

  void addEntries(PetscInt m, PetscInt const* idxm, PetscInt n, PetscInt const* idxn, Number const* values) {
    if(Number::getTape().isActive()) {
      int boundary_start = (int)rhs_data.size();

      for (int cur_row = 0; cur_row < m; cur_row += 1) {
        if(idxm[cur_row] < 0) { continue; }

        for (int cur_col = 0; cur_col < n; cur_col += 1) {
          if(idxn[cur_col] < 0) { continue; }

          addEntry(idxm[cur_row], idxn[cur_col], values[cur_row * n + cur_col]);
        }
      }

      if(boundary_start != (int)rhs_data.size()) {
        step_boundaries.push_back(boundary_start);

        Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&step_reverse, this, &no_delete));
      }
    }
  }

  void finalize(ADMat mat) {
    // TODO: Add collection of values that have been set by passive rhs and disable them here.
    Tape& tape = Number::getTape();

    step_boundaries.push_back((int)rhs_data.size());

    finalizeData(); // Sort the lhs_in_positions for better access

    comm_data.initialize(mat->mat, lhs_in_positions);

    if(comm_data.active && tape.isActive()) {
      std::vector<LhsInData> row_col_data(comm_data.total_request_recv);
      comm_data.commRowCol(lhs_in_positions.data(), row_col_data.data());

      std::vector<LhsInData> row_col_data_unique(row_col_data);
      lhs_out_positions.resize(comm_data.total_request_recv);

      std::sort(row_col_data_unique.begin(), row_col_data_unique.end());
      auto unique_end = std::unique(row_col_data_unique.begin(), row_col_data_unique.end());

      // Sanity check: Adjoint is not well formed if a value is set by multiple processors.
      if(INSERT_VALUES == mode) {
        if(unique_end != row_col_data_unique.end()) {
          AP_EXCEPTION("Value is set from multiple processors. Adjoint is not well defined.");
        }
      }

      // Delete non unqiue data
      row_col_data_unique.erase(unique_end, row_col_data_unique.end());

      int lhs_out_pos = 0;
      int cur_rank = 0;
      for(int i = 0; i < comm_data.total_request_recv; i += 1) {
        LhsInData const& cur = row_col_data[i];

        // Reset lhs_out_pos on rank shift
        while(i >= comm_data.recv_displs[cur_rank + 1]) {
          cur_rank += 1;
          lhs_out_pos = 0;
        }

        // Now search in the lhs_out values
        for(; lhs_out_pos < (int)row_col_data_unique.size(); lhs_out_pos += 1) {
          LhsInData const& cur_unique = row_col_data_unique[lhs_out_pos];
          if(cur.row == cur_unique.row && cur.col == cur_unique.col) {
            lhs_out_positions[i] = lhs_out_pos;
            break;
          }
        }
      }

      // Register all modified values
      lhs_identifiers.resize(row_col_data_unique.size());
      int pos = 0;
      for(LhsInData const& cur : row_col_data_unique) {
        auto func = [&](Wrapper& value) {

          if(tape.isActive()) {
            tape.registerExternalFunctionOutput(value);
            lhs_identifiers[pos] = value.getIdentifier();
          } else {
            tape.deactivateValue(value);
          }
        };
        PetscCallVoid(MatAccessValue(func, cur.row, cur.col, mat));

        pos += 1;
      }
    }

    if(comm_data.active && tape.isActive()) {
      Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&assemble_reverse, this, &ad_delete));
    } else {
      delete this;
    }
  }

  static void step_reverse(Tape* AP_U(tape), void* d, VectorInterface* va) {
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;
    int ad_vec_size = va->getVectorSize();

    data->adjoint_step -= 1;
    int from = data->step_boundaries[data->adjoint_step + 1];
    int to = data->step_boundaries[data->adjoint_step];

    for(int i = from - 1; i >= to; i -= 1) {
      va->updateAdjointVec(data->rhs_data[i].rhs_identifier, &data->adjoint_data[data->rhs_data[i].lhs_pos * ad_vec_size]);
      if(INSERT_VALUES == data->mode) {
        // Only the last (reverse the first) set variable gets the adjoint.
        for(int dataPos = 0; dataPos < ad_vec_size; dataPos += 1) {
          data->adjoint_data[data->rhs_data[i].lhs_pos * ad_vec_size + dataPos] = Real();
        }
      }
    }

    if(data->adjoint_step == 0) {
      data->adjoint_data.resize(0);
    }
  }

  static void assemble_reverse(Tape* AP_U(tape), void* d, VectorInterface* va) {
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;

    data->adjoint_step = (int)data->step_boundaries.size() - 1;

    data->communicateADData(va);
  }

  void communicateADData(VectorInterface* va) {
    int vec_size = va->getVectorSize();
    std::vector<Real> adjoints_recv(comm_data.total_request_recv * vec_size);
    for(int i = 0; i < comm_data.total_request_recv; i += 1) {
      Identifier identifier = lhs_identifiers[lhs_out_positions[i]];
      va->getAdjointVec(identifier, &adjoints_recv[i * vec_size]);
    }

    // Step 4: Clean adjoints
    if(INSERT_VALUES == mode) {
      for(Identifier const& cur : lhs_identifiers) {
        va->resetAdjointVec(cur);
      }
    }

    // Step 5: Send requested adjoint values to ranks
    MPI_Datatype adjoint_type = MPI_DOUBLE;
    if( 1 != vec_size) {
      MPI_Type_contiguous(vec_size, MPI_DOUBLE, &adjoint_type);
      MPI_Type_commit(&adjoint_type);
    }

    adjoint_data.resize(comm_data.total_request_send * vec_size);
    comm_data.commReverse(adjoints_recv.data(), adjoint_data.data(), adjoint_type);

    if( 1 != vec_size) {
      MPI_Type_free(&adjoint_type);
    }
  }

  static void no_delete(Tape* tape, void* data) {
    (void)tape;
    (void)data;
  }

  static void ad_delete(Tape* tape, void* d) {
    (void)tape;
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;

    delete data;
  }


  private:
  void addEntry(PetscInt row, PetscInt col, Number const& rhs_value) {
    if(Number::getTape().isIdentifierActive(rhs_value.getIdentifier())) {
      PetscInt lhs_cache_pos = insertLhsPos(row, col);

      rhs_data.push_back({lhs_cache_pos, rhs_value.getIdentifier()});
    }
  }

  PetscInt insertLhsPos(PetscInt row, PetscInt col) {
    // Search for the position.
    for(size_t i = 0; i < lhs_in_positions.size(); i += 1) {
      if(row == lhs_in_positions[i].row && col == lhs_in_positions[i].col) {
        return i;
      }
    }

    // Did not find the position add it.
    lhs_in_positions.push_back({row, col});

    return (PetscInt)lhs_in_positions.size() - 1;
  }

  void finalizeData() {
    // We sort an index array, which gives us the lookup to update the stored positions in the rhs entries.
    std::vector<PetscInt> lookup(lhs_in_positions.size());
    PetscInt i = 0;
    for(PetscInt& cur : lookup) { cur = i; i += 1; }

    // Access the actual data for comparison.
    auto compare = [&](PetscInt a_i, PetscInt b_i) -> bool {
      LhsInData const& a = lhs_in_positions[a_i];
      LhsInData const& b = lhs_in_positions[b_i];

      return a < b;

    };

    // Do the sort.
    std::sort(lookup.begin(), lookup.end(), compare);

    // Now 'sort' the lhs data
    std::vector<LhsInData> copy = lhs_in_positions;
    for(size_t i = 0; i < copy.size(); i += 1) {
      lhs_in_positions[i] = copy[lookup[i]];
    }

    // Build the reverse lookup table
    std::vector<PetscInt> lookup_reverse(lhs_in_positions.size());
    i = 0;
    for(PetscInt& cur : lookup) {
      lookup_reverse[cur] = i;
      i += 1;
    }

    // Update the lhs positions for the rhs data.
    for(RhsData& cur : rhs_data) {
      cur.lhs_pos = lookup_reverse[cur.lhs_pos];
    }
  }
};


PetscErrorCode MatAssemblyBegin(ADMat mat, MatAssemblyType type) {
  return MatAssemblyBegin(mat->mat, type);
}

PetscErrorCode MatAssemblyEnd(ADMat mat, MatAssemblyType type) {
  PetscCall(MatAssemblyEnd(mat->mat, type));

  ADMatCreateADData(mat);

  if(nullptr == mat->transaction_data) {
    mat->transaction_data = new ADData_MatSetValues(INSERT_VALUES);
  }

  ADData_MatSetValues* data = reinterpret_cast<ADData_MatSetValues*>(mat->transaction_data);

  data->finalize(mat);
  mat->transaction_data = nullptr;

  return PETSC_SUCCESS;
}
PetscErrorCode MatConvert(ADMat mat, MatType newtype, MatReuse reuse, ADMat *M) {
  AP_UNUSED(mat, newtype, reuse, M);

  throw std::runtime_error("'MatConvert' not implemented.");

  return PETSC_SUCCESS;
}

PetscErrorCode MatCreate(MPI_Comm comm, ADMat* mat) {
  (*mat) = new ADMatImpl();
  PetscCall(MatCreate(comm, &(*mat)->mat));

  return PETSC_SUCCESS;
}

PetscErrorCode MatCreateAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], ADMat *A) {
  (*A) = new ADMatImpl();
  PetscCall(MatCreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, &(*A)->mat));

  return PETSC_SUCCESS;
}

PetscErrorCode MatCreateDense(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Number *data, ADMat *A) {
  AP_UNUSED(comm, m, n, M, N, data, A);

  std::cout << "Not yet supported." << std::endl;
  return PETSC_ERR_ARG_WRONGSTATE;
}

PetscErrorCode MatDestroy(ADMat* mat) {

  if(nullptr != (*mat)->mat_i) {
    Tape& tape = Number::getTape();
    MatIterateAllEntries([&](PetscInt AP_U(row), PetscInt AP_U(col), Wrapper& el) {
      tape.deactivateValue(el);
    }, *mat);

    delete (*mat)->mat_i;
  }

  PetscCall(MatDestroy(&(*mat)->mat));
  delete *mat;
  *mat = nullptr;

  return PETSC_SUCCESS;
}

PetscErrorCode MatDuplicate(ADMat mat, MatDuplicateOption op, ADMat* newv) {
  *newv = new ADMatImpl();

  PetscCall(MatDuplicate(mat->mat, op, &(*newv)->mat));

  ADMatCreateADData(*newv);

  if(op == MAT_COPY_VALUES) {
    auto func = [&] (PetscInt AP_U(row), PetscInt AP_U(col), Wrapper& a, Wrapper& b) {
      b = a;
    };
    PetscCall(MatIterateAllEntries(func, mat, *newv));
  }

  return PETSC_SUCCESS;

}

PetscErrorCode MatGetInfo(ADMat mat, MatInfoType flag, MatInfo *info) {
  return MatGetInfo(mat->mat, flag, info);
}

PetscErrorCode MatGetLocalSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetLocalSize(mat->mat, m, n);
}

PetscErrorCode MatGetOwnershipRange(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetOwnershipRange(mat->mat, m, n);
}

PetscErrorCode MatGetSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetSize(mat->mat, m, n);
}

PetscErrorCode MatGetType                (ADMat mat, MatType *type) {
  return MatGetType(mat->mat, type);
}

PetscErrorCode MatGetValues(ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], Number v[]) {
  for(PetscInt row = 0; row < m; row += 1) {
    if(idxm[row] < 0) { continue; }

    for(PetscInt col = 0; col < n; col += 1) {
      if (idxn[col] < 0) { continue; }

      auto func = [&] (Wrapper& wrap) {
        v[row * n + col] = wrap;
      };
      PetscCall(MatAccessValue(func, idxm[row], idxn[col], mat));
    }
  }

  return PETSC_SUCCESS;
}

PetscErrorCode MatGetValue(ADMat mat, PetscInt row, PetscInt col, Number* v) {
  return MatGetValues(mat, 1, &row, 1, &col, v);
}

PetscErrorCode MatMPIAIJSetPreallocation(ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]) {
  return MatMPIAIJSetPreallocation(B->mat, d_nz, d_nnz, o_nz, o_nnz);
}

struct ADData_MatMult : public ReverseDataBase<ADData_MatMult> {

  Mat            A_v;
  ADMatData*     A_i;
  Vec            x_v;
  AdjointVecData x_i;
  AdjointVecData y_i;

  ADData_MatMult(ADMat A, ADVec x, ADVec y) : A_v(), A_i(), x_v(), x_i(x), y_i(y) {
    PetscCallVoid(VecDuplicate(x->vec, &x_v));
    PetscCallVoid(VecCopy(x->vec, x_v));

    ADMatCopyForReverse(A, &A_v, &A_i);
  }

  ~ADData_MatMult() {
    PetscCallVoid(VecDestroy(&x_v));
    PetscCallVoid(MatDestroy(&A_v));
    delete A_i;
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    Mat A_b;
    PetscCallVoid(x_i.createAdjoint());
    PetscCallVoid(y_i.createAdjoint());
    PetscCallVoid(MatDuplicate(A_v, MAT_SHARE_NONZERO_PATTERN, &A_b));

    DyadicProductHelper dyadic = {};
    dyadic.init(A_v, x_v);
    PetscCallVoid(dyadic.communicateValues(x_v));

    int dim = vi->getVectorSize();


    int cur_dim = 0;
    auto dyadic_update = [&] (PetscInt row, PetscInt col, Real& AP_U(value), Identifier& id) {
      Real entry_y_b;
      Real entry_x_v;
      PetscCallVoid(VecGetValues(y_i.getVec(), 1, &row, &entry_y_b));
      PetscCallVoid(dyadic.getValue(x_v, col, &entry_x_v));

      if(tape->isIdentifierActive(id)) {
        vi->updateAdjoint(id, cur_dim, entry_y_b * entry_x_v);
      }
    };

    for(; cur_dim < dim; cur_dim += 1) {
      y_i.getAdjoint(vi, cur_dim);

      PetscCallVoid(MatMultTranspose(A_v, y_i.getVec(), x_i.getVec()));
      x_i.updateAdjoint(vi, cur_dim);

      PetscCallVoid(MatIterateAllEntries(dyadic_update, A_v, A_i));
    }

    PetscCallVoid(x_i.freeAdjoint());
    PetscCallVoid(y_i.freeAdjoint());

    PetscCallVoid(MatDestroy(&A_b));
  }
};

PetscErrorCode MatMult(ADMat mat, ADVec x, ADVec y) {
  PetscCall(MatMult(mat->mat, x->vec, y->vec));

  bool active_mat;
  bool active_x;
  ADMatIsActive(mat, &active_mat);
  ADVecIsActive(x, &active_x);
  bool active_y = active_mat || active_x;

  if(active_y) {
    ADVecRegisterExternalFunctionOutput(y);

    ADData_MatMult* data = new ADData_MatMult(mat, x, y);
    data->push();
  } else {
    ADVecMakePassive(y);
  }

  return PETSC_SUCCESS;
}

struct ADData_MatNorm : public ReverseDataBase<ADData_MatNorm> {

  Mat           x_v;
  ADMatData*    x_i;
  NormType      type;
  Real          v_v;
  Identifier    v_i;

  ADData_MatNorm(ADMat x, NormType type, Number* v) : x_v(), x_i(), type(type), v_v(v->getValue()), v_i() {

    Number::getTape().registerExternalFunctionOutput(*v);
    v_i = v->getIdentifier();

    ADMatCopyForReverse(x, &x_v, &x_i);
  }

  ~ADData_MatNorm() {
    PetscCallVoid(MatDestroy(&x_v));
    delete x_i;
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    int dim = vi->getVectorSize();

    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)x_v, &comm);

    // Get the lhs adjoint.
    std::vector<Real> v_b(dim);
    vi->getAdjointVec(v_i, v_b.data());
    vi->resetAdjointVec(v_i);
    MPI_Allreduce(MPI_IN_PLACE, v_b.data(), dim, MPI_DOUBLE, MPI_SUM, comm);

    if(type == NORM_1 || type == NORM_INFINITY) {
      std::set<PetscInt> selected = {};

      // First select the rows/cols with the maximum values.
      if(type == NORM_1) {
        PetscInt col_size;
        PetscInt row_size;
        PetscCallVoid(MatGetSize(x_v, &row_size, &col_size));


        std::vector<Real> col_sums(col_size);
        PetscCallVoid(MatGetColumnSumAbs(x_v, col_sums.data()));

        PetscInt cur_col = 0;
        for(Real const& value_col : col_sums) {
          if(value_col == v_v) {
            selected.insert(cur_col);
          }
          cur_col += 1;
        }
      }
      else if(type == NORM_INFINITY) {
        Vec row_sums;
        PetscCallVoid(MatCreateVecs(x_v, &row_sums, NULL));
        PetscCallVoid(MatGetRowSumAbs(x_v, row_sums));

        auto func = [&](PetscInt row, Real& value) {
          if(value == v_v) {
            selected.insert(row);
          }
        };
        VecIterateAllEntries(func, row_sums);

        PetscCallVoid(VecDestroy(&row_sums));
      }

      // Iterate and perform the adjoint update.
      auto func = [&](PetscInt row, PetscInt col, Real& value, Identifier& id) {
        PetscInt sel = type == NORM_1 ? col : row;
        if(selected.end() == selected.find(sel)) {
          return;
        }

        Real jac = 0.0;
        if(value < 0.0) {
          jac = -1.0;
        }
        else if(value > 0.0) {
          jac = 1.0;
        }

        for(int i = 0; i < dim; i += 1) {
          if( tape->isIdentifierActive(id)) {
            vi->updateAdjoint(id, i, v_b[i] * jac);
          }
        }
      };
      MatIterateAllEntries(func, x_v, x_i);
    }
    else if(type == NORM_FROBENIUS) {
      auto func = [&](PetscInt AP_U(row), PetscInt AP_U(col), Real& value, Identifier& id) {
        for(int i = 0; i < dim; i += 1) {
          if(tape->isIdentifierActive(id)) {
            vi->updateAdjoint(id, i, v_b[i] * value / v_v);
          }
        }
      };
      MatIterateAllEntries(func, x_v, x_i);
    }
    else {
      AP_EXCEPTION("Reverse of norm kind %d is not implemented.", type);
    }
  }
};

PetscErrorCode MatNorm(ADMat x, NormType type, Number *val) {
  PetscCall(MatNorm(x->mat, type, &val->value()));

  ADData_MatNorm* data = new ADData_MatNorm(x, type, val);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode MatSeqAIJSetPreallocation (ADMat B, PetscInt nz, const PetscInt nnz[]) {
  return MatSeqAIJSetPreallocation(B->mat, nz, nnz);
}

PetscErrorCode MatSetFromOptions(ADMat mat) {
  return MatSetFromOptions(mat->mat);
}

PetscErrorCode MatSetOption(ADMat x, MatOption op, PetscBool flag) {
  return MatSetOption(x->mat, op, flag);
}

PetscErrorCode MatSetSizes(ADMat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N) {
  PetscCall(MatSetSizes(mat->mat, m, n, M, N));

  return PETSC_SUCCESS;
}

PetscErrorCode MatSetValues(ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const Number v[], InsertMode addv) {
  std::vector<Real> v_p(m * n);
  for(size_t i = 0; i < v_p.size(); i += 1) {
    v_p[i] = v[i].getValue();
  }
  PetscCall(MatSetValues(mat->mat, m, idxm, n, idxn, v_p.data(), addv));

  ADData_MatSetValues* data = nullptr;
  if(nullptr == mat->transaction_data) {
    data = new ADData_MatSetValues(addv);
    mat->transaction_data = data;
  }
  else {
    data = (ADData_MatSetValues*)mat->transaction_data;
  }

  data->addEntries(m, idxm, n, idxn, v);

  return PETSC_SUCCESS;
}

PetscErrorCode MatSetValue(ADMat mat, PetscInt i, PetscInt j, Number v, InsertMode addv) {
  return MatSetValues(mat, 1, &i, 1, &j, &v, addv);
}

PetscErrorCode MatView(ADMat mat, PetscViewer viewer) {
  return MatView(mat->mat, viewer);
}

PetscErrorCode MatZeroEntries(ADMat mat) {
  auto func = [&] (PetscInt AP_U(row), PetscInt AP_U(col), Wrapper& value) {
    value = Real();
  };
  return MatIterateAllEntries(func, mat);
}

PetscErrorCode MatSeqAIJGetEntrySize(Mat mat, PetscInt* entries) {
  PetscInt const* row_offset;
  PetscInt low;
  PetscInt high;
  PetscInt row_range;

  PetscCall(MatGetOwnershipRange(mat, &low, &high));
  PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, nullptr, nullptr, nullptr));

  row_range = high - low;
  *entries = row_offset[row_range];

  return PETSC_SUCCESS;
}

PetscErrorCode MatMPIAIJGetEntrySize(Mat mat, PetscInt* diag_entries, PetscInt* off_diag_entries) {
  Mat matd;
  Mat mato;
  PetscInt const* colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

  PetscCall(MatSeqAIJGetEntrySize(matd, diag_entries));
  PetscCall(MatSeqAIJGetEntrySize(mato, off_diag_entries));

  return PETSC_SUCCESS;
}

/*-------------------------------------------------------------------------------------------------
 * AD specific functions
 */

void ADMatCreateADData(ADMat mat) {
  if(mat->mat_i != nullptr) {
    return;
  }

  MatType ptype;
  PetscCallVoid(MatGetType(mat->mat, &ptype));

  ADMatType type = ADMatDataPTypeToEnum(ptype);

  if(type == ADMatType::ADMatMPIAIJ) {
    PetscInt diag_size;
    PetscInt off_diag_size;

    PetscCallVoid(MatMPIAIJGetEntrySize(mat->mat, &diag_size, &off_diag_size));
    mat->mat_i = new ADMatMPIAIJData(diag_size, off_diag_size);
  }
  else if(type == ADMatType::ADMatSeqAIJ){
    PetscInt size;

    PetscCallVoid(MatSeqAIJGetEntrySize(mat->mat, &size));
    mat->mat_i = new ADMatSeqAIJData(size);
  } else {
    AP_EXCEPTION("Unsupported matrix type %d.", (int)type);
  }
}

void ADMatCopyForReverse(ADMat mat, Mat* newm, ADMatData** newd) {

  if(nullptr != newm) {
    PetscCallVoid(MatDuplicate(mat->mat, MAT_COPY_VALUES, newm));
  }

  if(nullptr != newd) {
    *newd = mat->mat_i->clone();
  }
}

void ADMatIsActive(ADMat mat, bool* a) {
  Tape& tape = Number::getTape();

  int active = 0;
  auto func = [&](PetscInt AP_U(row), PetscInt AP_U(col), Wrapper& value) {
    active += tape.isIdentifierActive(value.getIdentifier());
  };
  PetscCallVoid(MatIterateAllEntries(func, mat));

  MPI_Comm comm;
  PetscCallVoid(PetscObjectGetComm((PetscObject)mat->mat, &comm));

  MPI_Allreduce(MPI_IN_PLACE, &active, 1, MPI_INTEGER, MPI_SUM, comm);
  *a = 0 != active;
}

/*-------------------------------------------------------------------------------------------------
 * Debug functions
 */

struct MatrixEntry {
  PetscInt row;
  PetscInt col;
  Real value;
  Identifier id;

  void print(std::ostream& out) {
    if(ADPetscOptionsGetDebugOutputIdentifiers()) {
      out << row << " " << col << " " << value << "(" << debugOutputId(id) << ")\n";
    } else {
      out << row << " " << col << " " << value << "\n";
    }
  }
};

bool operator<(MatrixEntry const& a, MatrixEntry const& b) {
  return a.row < b.row || (a.row == b.row && a.col < b.col);
}

PetscErrorCode ADMatDebugOutputImpl(Mat mat_v, ADMatData* mat_i, std::string m, int id, bool forward, VectorInterface* vi) {
  std::ostream& out = ADPetscOptionsGetDebugOutputStream();
  out.setf(std::ios::scientific);
  out.setf(std::ios::showpos);
  out.precision(ADPetscOptionsGetDebugOutputPrecission());

  PetscInt M, N;
  MPI_Comm comm;
  int rank;
  int size;
  PetscCall(MatGetSize(mat_v, &M, &N));
  PetscCall(PetscObjectGetComm((PetscObject)mat_v, &comm));
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int vec_size = 1; // Primal has a vector size of one.
  if(!forward) {
    vec_size = vi->getVectorSize();
  }

  if(rank == 0 || ADPetscOptionsGetDebugOutputObjInfoOnAllRanks()) {
    if(forward) {
      out << m << " forward matrix id: " << id << std::endl;
    } else {
      out << m << " reverse matrix id: " << id << std::endl;
    }
    out << "Matrix of global size " << M << "x" << N << std::endl;
  }
  MPI_Barrier(comm);
  for(int cur_dim = 0; cur_dim < vec_size; cur_dim += 1) {
    for(int cur_rank = 0; cur_rank < size; cur_rank += 1) {
      if(cur_rank == rank) {
        std::vector<MatrixEntry> data = {};
        if(forward) {
          out << "Rank: " << cur_rank << "\n";
          auto func = [&](PetscInt row, PetscInt col, Real& value, Identifier& id) {
            data.push_back(MatrixEntry({row, col, value, id}));
          };

          MatIterateAllEntries(func, mat_v, mat_i);
        } else {
          out << "Rank: " << cur_rank << " dim: " << cur_dim <<"\n";
          auto func = [&](PetscInt row, PetscInt col, Real& AP_U(value), Identifier& id) {
            data.push_back(MatrixEntry({row, col, vi->getAdjoint(id, cur_dim), id}));
          };

          MatIterateAllEntries(func, mat_v, mat_i);
        }

        std::sort(data.begin(), data.end());
        for(MatrixEntry& cur : data) {
          cur.print(out);
        }
        out.flush();
      }
      MPI_Barrier(comm);
    }
  }

  return PETSC_SUCCESS;
}

struct ADData_MatDebugOutput : public ReverseDataBase<ADData_MatDebugOutput> {

  Mat mat_v;
  ADMatData* mat_i;
  std::string m;
  int id;

  ADData_MatDebugOutput(ADMat mat, std::string m, int id) : mat_v(), mat_i(), m(m), id(id) {
    ADMatCopyForReverse(mat, &mat_v, &mat_i);
  }

  ~ADData_MatDebugOutput() {
    MatDestroy(&mat_v);
    delete mat_i;
  }

  void reverse(Tape* AP_U(tape), VectorInterface* vi) {
    ADMatDebugOutputImpl(mat_v, mat_i, m, id, false, vi);
  }
};

void ADMatDebugOutput(ADMat mat, std::string m, int id) {
  if(ADPetscOptionsGetDebugOutputPrimal()) {
    ADMatDebugOutputImpl(mat->mat, mat->mat_i, m, id, true, nullptr);
  }
  if(ADPetscOptionsGetDebugOutputReverse()) {
    ADData_MatDebugOutput* data = new ADData_MatDebugOutput(mat, m, id);
    data->push();
  }
}
void ADMatDebugOutput(Mat mat, std::string m, int id) {
  (void)mat;
  (void)m;
  (void)id;
}

AP_NAMESPACE_END
