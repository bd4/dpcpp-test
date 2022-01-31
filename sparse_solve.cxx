#include <time.h>

#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <typeinfo>

#define NRUNS 10

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
//#include "oneapi/mkl/spblas.hpp"

using index_t = std::int64_t;
using spindex_t = std::int32_t;

#ifdef OLD_NAME
#define getrfnp_batch_strided_scratchpad_size getrfnp_batch_scratchpad_size
#define getrfnp_batch_strided getrfnp_batch
#define getrf_batch_strided_scratchpad_size getrf_batch_scratchpad_size
#define getrf_batch_strided getrf_batch
#endif

template <typename T>
inline void read_carray(std::ifstream &f, int n, std::complex<T> *Adata) {
    for (int i = 0; i < n; i++) {
        // std::cout << i << " " << std::endl;
        f >> Adata[i];
    }
}

inline void read_iarray(std::ifstream &f, int n, int *data) {
    for (int i = 0; i < n; i++) {
        f >> data[i];
    }
}

template <typename T>
struct sycl_spmatrix {
    spindex_t nnz;
    spindex_t nrows;
    spindex_t ncols;
    spindex_t *row_ptr;
    spindex_t *col_ind;
    T *values;
};

template <typename T>
struct sycl_spmatrix_lu {
    struct sycl_spmatrix<T> L;
    struct sycl_spmatrix<T> U;
};

template <typename T>
struct sycl_spmatrix_lu<T> create_batched_lu_spmatrix(
    sycl::queue q, index_t nrows, index_t ncols, index_t batch_size, T *data) {
    struct sycl_spmatrix_lu<T> lu;

    lu.L.nnz = 0;
    lu.U.nnz = 0;
    int idx = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < ncols; j++) {
            for (int i = 0; i < nrows; i++) {
                idx = b * nrows * ncols + j * nrows + i;
                if (data[idx] != 0) {
                    if (i < j) {
                        lu.L.nnz++;
                    } else {
                        lu.U.nnz++;
                    }
                }
            }
        }
    }

    std::cout << "L nnz " << lu.L.nnz << std::endl;
    std::cout << "U nnz " << lu.U.nnz << std::endl;

    lu.L.nrows = lu.U.nrows = nrows * batch_size;
    lu.U.ncols = lu.U.ncols = ncols * batch_size;

    lu.L.values = sycl::malloc_shared<T>(lu.L.nnz, q);
    lu.L.col_ind = sycl::malloc_shared<spindex_t>(lu.L.nnz, q);
    lu.L.row_ptr = sycl::malloc_shared<spindex_t>(lu.L.nrows + 1, q);
    lu.U.values = sycl::malloc_shared<T>(lu.U.nnz, q);
    lu.U.col_ind = sycl::malloc_shared<spindex_t>(lu.U.nnz, q);
    lu.U.row_ptr = sycl::malloc_shared<spindex_t>(lu.U.nrows + 1, q);

    index_t L_val_i = 0;
    index_t L_row_i = 0;
    index_t U_val_i = 0;
    index_t U_row_i = 0;
    bool L_row_set;
    bool U_row_set;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < nrows; i++) {
            L_row_set = false;
            U_row_set = false;
            for (int j = 0; j < ncols; j++) {
                T value = data[b * nrows * ncols + j * nrows + i];
                if (value != 0) {
                    if (i < j) {
                        lu.L.values[L_val_i] = value;
                        lu.L.col_ind[L_val_i] = j;
                        if (!L_row_set) {
                            lu.L.row_ptr[i] = L_val_i;
                            L_row_set = true;
                        }
                        L_val_i++;
                    } else {
                        lu.U.values[U_val_i] = value;
                        lu.U.col_ind[U_val_i] = j;
                        if (!U_row_set) {
                            lu.U.row_ptr[i] = U_val_i;
                            U_row_set = true;
                        }
                        U_val_i++;
                    }
                }
            }
        }
    }
    lu.L.row_ptr[nrows] = lu.L.nnz;
    lu.U.row_ptr[nrows] = lu.U.nnz;

    return lu;
}

template <typename T>
void free_batched_spmatrix(sycl::queue q, struct sycl_spmatrix<T> spmat) {
    sycl::free(spmat.row_ptr, q);
    sycl::free(spmat.col_ind, q);
    sycl::free(spmat.values, q);
}

template <typename T>
void test(cl::sycl::queue q, index_t n = 140, index_t nrhs = 1,
          index_t batch_size = 384) {
    index_t lda, ldb;
    int Aptr_count, Bptr_count, Adata_count, Bdata_count, piv_count;
    T **d_Aptr, **d_Bptr, **h_Aptr, **h_Bptr;
    T *h_Adata, *d_Adata, *h_Bdata, *d_Bdata;
    index_t **d_piv_ptr, **h_piv_ptr;
    index_t *h_piv, *d_piv;

    using CT = std::complex<T>;

#ifdef READ_INPUT
    std::ifstream f("zgetrs.txt", std::ifstream::in);

    f >> n;
    f >> nrhs;
    f >> lda;
    f >> ldb;
    f >> batch_size;

#else
    lda = n;
    ldb = n;
#endif

    std::cout << "n    = " << n << std::endl;
    std::cout << "nrhs = " << nrhs << std::endl;
    std::cout << "lda  = " << lda << std::endl;
    std::cout << "ldb  = " << ldb << std::endl;
    std::cout << "batch_size = " << batch_size << std::endl;

    Aptr_count = Bptr_count = batch_size;
    Adata_count = n * n * batch_size;
    Bdata_count = n * nrhs * batch_size;
    piv_count = n * batch_size;

    std::cout << "Aptr count  = " << Aptr_count << std::endl;
    std::cout << "Bptr count  = " << Bptr_count << std::endl;
    std::cout << "Adata count = " << Adata_count << std::endl;
    std::cout << "Bdata count = " << Bdata_count << std::endl;
    std::cout << "piv_count   = " << piv_count << std::endl;

    h_Adata = cl::sycl::malloc_host<T>(Adata_count, q);
    h_Bdata = cl::sycl::malloc_host<T>(Bdata_count, q);
    d_Adata = cl::sycl::malloc_device<T>(Adata_count, q);
    d_Bdata = cl::sycl::malloc_device<T>(Bdata_count, q);

    h_piv = cl::sycl::malloc_host<index_t>(piv_count, q);
    d_piv = cl::sycl::malloc_device<index_t>(piv_count, q);

    // pointers to the groups for group API
    d_Aptr = cl::sycl::malloc_device<T *>(Aptr_count, q);
    d_Bptr = cl::sycl::malloc_device<T *>(Bptr_count, q);
    d_piv_ptr = cl::sycl::malloc_device<index_t *>(batch_size, q);
    h_Aptr = cl::sycl::malloc_host<T *>(Aptr_count, q);
    h_Bptr = cl::sycl::malloc_host<T *>(Bptr_count, q);
    h_piv_ptr = cl::sycl::malloc_host<index_t *>(batch_size, q);

    std::cout << "malloc done" << std::endl;

    std::ostringstream ss;
    std::string run_label;

    ss << "n=" << n << ";nrhs=" << nrhs << ";batches=" << batch_size
       << ";t=" << typeid(T).name();
    run_label = ss.str();

    q.fill<T>(h_Adata, T(0.0), Adata_count);
    q.fill<T>(h_Bdata, T(0.0), Bdata_count);
    q.wait();

    // 2.0 on the anti diagonal, fails without pivot
    /*
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_Adata[b * n * n + i * n + (n - 1 - i)] = T(2.0);
        }
    }
    */

    // 2's on diag, 1's in last row (except for bottom right which is 2).
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_Adata[b * n * n + i * n + i] = T(2.0);
        }
        /*
        for (int i = 0; i < n - 1; i++) {
            h_Adata[b * n * n + i * n + n - 1] = T(1.0);
        }
        */
    }

    // for collecting pointers to pointers
    for (int i = 0; i < batch_size; i++) {
        h_Aptr[i] = d_Adata + (n * n * i);
        h_Bptr[i] = d_Bdata + (n * nrhs * i);
        h_piv_ptr[i] = d_piv + (i * n);
    }

    std::cout << "read/init done" << std::endl;

    q.copy(h_Aptr, d_Aptr, Aptr_count);
    q.copy(h_Adata, d_Adata, Adata_count);
    q.copy(h_Bptr, d_Bptr, Bptr_count);
    q.copy(h_Bdata, d_Bdata, Bdata_count);
    q.copy(h_piv_ptr, d_piv_ptr, batch_size);
    q.copy(h_piv, d_piv, piv_count);

    // clear out host response
    q.fill<T>(h_Bdata, T(0.0), Bdata_count);

    q.wait();

    std::cout << "memcpy done" << std::endl;

    struct timespec start, end;
    double elapsed, total = 0.0;
    double total_strided = 0.0;
    int *info, info_sum;

    auto trans_op = oneapi::mkl::transpose::nontrans;

#if defined(NOPIVOT)
    auto scratch_countf =
        oneapi::mkl::lapack::getrfnp_batch_strided_scratchpad_size<T>(
            q, n, n, lda, n * n, batch_size);
    auto scratchf = cl::sycl::malloc_device<T>(scratch_countf, q);
#elif defined(GROUP_API_NOPIVOT)
    auto scratch_countf = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<T>(
        q, &n, &n, &lda, 1, &batch_size);
    auto scratchf = cl::sycl::malloc_device<T>(scratch_countf, q);
#else
    auto scratch_countf =
        oneapi::mkl::lapack::getrf_batch_strided_scratchpad_size<T>(
            q, n, n, lda, n * n, n, batch_size);
    auto scratchf = cl::sycl::malloc_device<T>(scratch_countf, q);
#endif

    // LU factorize first
    try {
#if defined(NOPIVOT)
        auto e2 = oneapi::mkl::lapack::getrfnp_batch_strided(
            q, n, n, d_Adata, lda, n * n, batch_size, scratchf, scratch_countf);
        e2.wait_and_throw();
#elif defined(GROUP_API_NOPIVOT)
        auto e2 = oneapi::mkl::lapack::getrfnp_batch(
            q, &n, &n, h_Aptr, &lda, 1, &batch_size, scratchf, scratch_countf);
        e2.wait_and_throw();
#else
        // getrf_batch_strided: USM API
        auto e2 = oneapi::mkl::lapack::getrf_batch_strided(
            q, n, n, d_Adata, lda, n * n, d_piv, n, batch_size, scratchf,
            scratch_countf);
        e2.wait_and_throw();
#endif
    } catch (oneapi::mkl::lapack::invalid_argument const &e) {
        std::cerr << e.what() << " arg #: " << e.info() << std::endl;
        std::cerr << e.detail() << std::endl;
    } catch (...) {
        std::cout << "Still threw" << std::endl;
    }

    q.copy(d_Adata, h_Adata, Adata_count);
    q.copy(d_piv, h_piv, piv_count);
    q.wait();

    // check result
    T err = T(0.0);
    T correct_value = T(0.0);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    correct_value = T(2.0);
                } else if (j < n - 1 && i == n - 1) {
                    correct_value = T(0.0);
                } else {
                    correct_value = T(0.0);
                }
                err = h_Adata[b * n * n + i * n + j] - correct_value;
                if (std::abs(err) > 0.0) {
                    std::cout << "A ERROR of " << err << " at [" << b << ", "
                              << i << ", " << j << "]" << std::endl;
                    exit(1);
                }
            }
        }
    }

    // check pivot
#if !defined(NOPIVOT) && !defined(GROUP_API_NOPIVOT)
    index_t correct_value_int = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            correct_value_int = i + 1;
            err = h_piv[b * n + i] - correct_value_int;
            if (std::abs(err) > 0.0) {
                std::cout << "IPIV ERROR of " << err << " at [" << b << ", "
                          << i << "]" << std::endl;
                exit(1);
            }
        }
    }
#endif

    // check result
    /*
    T err = T(0.0);
    T correct_value = T(0.0);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                correct_value = T(0.0);
                if (i == j) correct_value = T(2.0);
                err = h_Adata[b * n * n + i * n + j] - correct_value;
                if (std::abs(err) > 0.0) {
                    std::cout << "A ERROR of " << err << " at [" << b << ", "
                              << i << ", " << j << "]" << std::endl;
                    exit(1);
                }
            }
        }
    }

    // check pivot
    index_t correct_value_int = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            correct_value_int = n - i;
            if (i >= n / 2) correct_value_int = i + 1;
            err = h_piv[b * n + i] - correct_value_int;
            if (std::abs(err) > 0.0) {
                std::cout << "IPIV ERROR of " << err << " at [" << b << ", "
                          << i << "]" << std::endl;
                exit(1);
            }
        }
    }
    */

    std::cout << "convert matrices to sparse" << std::endl;

    // solve
    auto splu = create_batched_lu_spmatrix(q, n, n, batch_size, h_Adata);

    oneapi::mkl::sparse::matrix_handle_t L_handle;
    oneapi::mkl::sparse::matrix_handle_t U_handle;

    oneapi::mkl::sparse::init_matrix_handle(&L_handle);
    oneapi::mkl::sparse::init_matrix_handle(&U_handle);

    oneapi::mkl::sparse::set_csr_data(
        L_handle, batch_size * n, batch_size * n, oneapi::mkl::index_base::zero,
        splu.L.row_ptr, splu.L.col_ind, splu.L.values);
    oneapi::mkl::sparse::set_csr_data(
        U_handle, batch_size * n, batch_size * n, oneapi::mkl::index_base::zero,
        splu.U.row_ptr, splu.U.col_ind, splu.U.values);

    auto b = sycl::malloc_shared<T>(batch_size * n, q);
    q.fill<T>(b, T(1.0), batch_size * n);
    q.wait();
    auto x1 = sycl::malloc_shared<T>(batch_size * n, q);
    auto x2 = sycl::malloc_shared<T>(batch_size * n, q);

    // Solve batched A * x = b in two steps with A = L U
    //   solve L * x1 = b
    //   solve U * x2 = x1
    std::cout << "back sub" << std::endl;
    oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::lower,
                              oneapi::mkl::transpose::nontrans,
                              oneapi::mkl::diag::unit, L_handle, b, x1);
    q.wait();
    std::cout << "forward sub" << std::endl;
    oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper,
                              oneapi::mkl::transpose::nontrans,
                              oneapi::mkl::diag::nonunit, U_handle, x1, x2);
    q.wait();

    // check solve result
    std::cout << "check" << std::endl;
    err = T(0.0);
    correct_value = T(0.0);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            correct_value = T(0.5);
            err = x2[b * n + i] - correct_value;
            if (std::abs(err) > 0.0) {
                std::cout << "A ERROR of " << err << " at [" << b << ", " << i
                          << "]" << std::endl;
                exit(1);
            }
        }
    }

    free_batched_spmatrix(q, splu.L);
    free_batched_spmatrix(q, splu.U);
    sycl::free(b, q);
    sycl::free(x1, q);
    sycl::free(x2, q);

    cl::sycl::free(d_Aptr, q);
    cl::sycl::free(d_Bptr, q);

    cl::sycl::free(h_Adata, q);
    cl::sycl::free(h_Bdata, q);
    cl::sycl::free(d_Adata, q);
    cl::sycl::free(d_Bdata, q);

    cl::sycl::free(d_piv_ptr, q);
    cl::sycl::free(h_piv, q);
    cl::sycl::free(d_piv, q);

    std::cout << "free done" << std::endl;
}

inline auto get_exception_handler() {
    static auto exception_handler = [](cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (cl::sycl::exception const &e) {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl;
                abort();
            }
        }
    };
    return exception_handler;
}

int main(int argc, char **argv) {
    auto q = cl::sycl::queue{get_exception_handler()};
    auto dev = q.get_device();
    std::string type;
    if (dev.is_cpu()) {
        type = "CPU  ";
    } else if (dev.is_gpu()) {
        type = "GPU  ";
    } else if (dev.is_host()) {
        type = "HOST ";
    } else {
        type = "OTHER";
    }
    std::cout << "[" << type << "] "
              << dev.get_info<cl::sycl::info::device::name>() << " {"
              << dev.get_info<cl::sycl::info::device::vendor>() << "}"
              << std::endl;

    index_t n = 10;
    index_t nrhs = 1;
    index_t batch_size = 2;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    if (argc > 2) {
        nrhs = std::stoi(argv[2]);
    }
    if (argc > 2) {
        batch_size = std::stoi(argv[3]);
    }

    // std::cout << "==== std::complex<float>  ====" << std::endl;
    // test<std::complex<float>>(q, n, nrhs, batch_size);
    std::cout << "==== float  ====" << std::endl;
    test<float>(q, n, nrhs, batch_size);
}
