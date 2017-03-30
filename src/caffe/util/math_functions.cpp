#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// Added by YanHan
// Add armadillo lib to perfrom some complex matrix operation
#include <armadillo>

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}


template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_symm<float>(const CBLAS_SIDE side,
    const CBLAS_UPLO uplo, const int M, const int N,
    const float alpha, const float* A, const float* B, const float beta,
    float* C){
  int lda = (side == CblasLeft) ? M : N;
  int ldb = (side == CblasRight) ? N : M;
  cblas_ssymm(CblasRowMajor, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void caffe_cpu_symm<double>(const CBLAS_SIDE side,
    const CBLAS_UPLO uplo, const int M, const int N,
    const double alpha, const double* A, const double* B, const double beta,
    double* C){
  int lda = (side == CblasLeft) ? M : N;
  int ldb = (side == CblasRight) ? N : M;
  cblas_dsymm(CblasRowMajor, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

// For symmtric matrix
template <>
void caffe_cpu_symv<float>(const int M,
    const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_ssymv(CblasRowMajor, CblasUpper, M, alpha, A, M, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_symv<double>(const int M,
    const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dsymv(CblasRowMajor, CblasUpper, M, alpha, A, M, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

// Added by YanHan
// Get the max value in the memory
template <>
float caffe_amax<float>(const int n, const float* a, int incx) {
	int idx = cblas_isamax(n, a, incx);
	return a[idx];
}
template <>
double caffe_amax<double>(const int n, const double* a, int incx) {
	int idx = cblas_idamax(n, a, incx);
	return a[idx];
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

// Added by YanHan
/* Since the following procedure need lapack which caffe does not included
 * We use another open source package armadillo instead, which is much more simple
 *
template <>
float caffe_cpu_det<float>(const int n, const float* m){
	// Since the LU decompose will overwrite the original memory
	// allocate new memory
	float* p = (float*)std::calloc(n, sizeof(float));
	int* ipiv = (int*)std::calloc(n, sizeof(int));
	// Copy the memory
	caffe_copy(n*sizeof(float), m, p);
	// Perform LU decompose
	int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, p, n, ipiv);
	// Check the result
	if (info > 0){
		// The U matrix is singular
		LOG(ERROR)<<"Singular Matrix detected, info: " << info;
	}else if(info<0){
		LOG(FATAL)<<"Failed to perfrom LU decompose, info: "<< info;
	}

	float res = 1;
	for (int i = 0; i < n; i++){
		int idx = i + i*n;
		if (ipiv[i] != i+1) {
			res = -res*p[idx];
		}else{
			res = res*p[idx];
		}
	}

	// Free the memor1
	std::free(p);
	std::free(ipiv);

	return res;
}

template <>
double caffe_cpu_det<double>(const int n, const double* m){
	// Since the LU decompose will overwrite the original memory
	// allocate new memory
	double* p = (double*)std::calloc(n, sizeof(double));
	int* ipiv = (int*)std::calloc(n, sizeof(int));
	// Copy the memory
	caffe_copy(n*sizeof(double), m, p);
	// Perform LU decompose
	int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, p, n, ipiv);
	// Check the result
	if (info > 0){
		// The U matrix is singular
		LOG(ERROR)<<"Singular Matrix detected, info: " << info;
	}else if(info<0){
		LOG(FATAL)<<"Failed to perfrom LU decompose, info: "<< info;
	}

	double res = 1;
	for (int i = 0; i < n; i++){
		int idx = i + i*n;
		if (ipiv[i] != i+1) {
			res = -res*p[idx];
		}else{
			res = res*p[idx];
		}
	}

	// Free the memor1
	std::free(p);
	std::free(ipiv);

	return res;
}
*/

template <>
float caffe_cpu_det<float>(const int n, const float* m){
	// Attention: The matrix in armadillo is column-major.
	// but in calc determinant, there is no difference
	arma::Mat<float> mm(m, n, n);
	return arma::det(mm);
}

template <>
double caffe_cpu_det<double>(const int n, const double* m){
	// Attention: The matrix in armadillo is column-major.
	// but in calc determinant, there is no difference
	arma::Mat<double> mm(m, n, n);
	return arma::det(mm);
}

// Added by YanHan
// Calc the inv of the matrix
template <>
void caffe_cpu_inv<float>(const int n, const float* m, float* m_inv){
	arma::Mat<float> mm(m, n, n);
	arma::Mat<float> mm_inv(m_inv, n, n, false);
	arma::inv(mm_inv, mm);
}

template <>
void caffe_cpu_inv<double>(const int n, const double* m, double* m_inv){
	arma::Mat<double> mm(m, n, n);
	arma::Mat<double> mm_inv(m_inv, n, n, false);
	arma::inv(mm_inv, mm);
}

template <>
void caffe_cpu_inv_sympd<float>(const int n, const float* m, float* m_inv){
	arma::Mat<float> mm(m, n, n);
	arma::Mat<float> mm_inv(m_inv, n, n, false);
	arma::inv_sympd(mm_inv, mm);
}

template <>
void caffe_cpu_inv_sympd<double>(const int n, const double* m, double* m_inv){
	arma::Mat<double> mm(m, n, n);
	arma::Mat<double> mm_inv(m_inv, n, n, false);
	arma::inv_sympd(mm_inv, mm);
}

// Added by YanHan
// Calc the sum instead of the absolute sum
template <>
float caffe_cpu_sum<float>(const int n, const float* x){
	// Here we use the loop directly, normally it is fast enough
	// but there must be a faster way to do it
	float accum = 0;
	for (int i = 0; i < n; i++){
		accum += *(x+i);
	}
	return accum; 
}
template <>
double caffe_cpu_sum<double>(const int n, const double* x){
	// Here we use the loop directly, normally it is fast enough
	// but there must be a faster way to do it
	double accum = 0;
	for (int i = 0; i < n; i++){
		accum += *(x+i);
	}
	return accum; 
}

}  // namespace caffe
