#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>
#include <vector>
#include "pde_types.h"
#include "expression_eval.h"
#include <cuda_runtime.h>

// Error handling
std::string CudaErrorToString(cudaError_t err);
bool CudaOk(cudaError_t err, std::string* error, const char* message);

// Device-side data structures
struct DeviceExpr {
  double constant;
  double x;
  double y;
  double z;

  __host__ __device__ double Eval(double px, double py, double pz = 0.0) const {
    return constant + x * px + y * py + z * pz;
  }
};

struct DeviceBC {
  int kind;
  DeviceExpr value;
  DeviceExpr alpha;
  DeviceExpr beta;
  DeviceExpr gamma;
};

struct DeviceBoundarySet {
  DeviceBC left;
  DeviceBC right;
  DeviceBC bottom;
  DeviceBC top;
};

struct DeviceNonlinearTerm {
  int kind;
  double coeff;
  int power;
};

// Conversion functions
DeviceExpr ToDeviceExpr(const BoundaryCondition::Expression& expr);
DeviceBC ToDeviceBC(const BoundaryCondition& bc);

// GPU Kernel Wrappers & Helpers
bool ApplyBoundaryKernels(double* d_grid, int nx, int ny,
                          const Domain& domain,
                          const DeviceBoundarySet& bc,
                          std::string* error);

// Krylov buffer management
struct CudaKrylovBuffers {
  int n = 0;
  int partial_count = 0;
  double* d_partials = nullptr;
  std::vector<double> h_partials;
};

bool AllocKrylovBuffers(int n, CudaKrylovBuffers* buf, std::string* error);
void FreeKrylovBuffers(CudaKrylovBuffers* buf);

// BLAS-like operations
bool DeviceDot(double* d_a, double* d_b, int n,
               CudaKrylovBuffers* buf,
               double* out, std::string* error);

bool DeviceNorm2(double* d_v, int n, CudaKrylovBuffers* buf, double* out, std::string* error);

void DeviceVecSet(double* d_v, double value, int n);
void DeviceVecCopy(const double* d_src, double* d_dst, int n);
void DeviceAxpy(double* d_y, const double* d_x, double a, int n);
void DeviceScale(double* d_v, double a, int n);
void DeviceLinComb2(double* d_out, const double* d_a, double alpha, const double* d_b, double beta, int n);

#endif // CUDA_UTILS_H
