#ifndef METAL_UTILS_H
#define METAL_UTILS_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pde_types.h"
#include <string>
#include <vector>

// Metal constants
constexpr int kReduceGroupSize = 256;

// Device-side data structures
struct DeviceExpr {
  float c0;
  float x;
  float y;
};

struct DeviceBC {
  int kind;
  DeviceExpr value;
  DeviceExpr alpha;
  DeviceExpr beta;
  DeviceExpr gamma;
};

struct DomainInfo {
  int nx, ny;
  float xmin, xmax, ymin, ymax;
  float dx, dy;
};

// Metal helper functions
bool MetalOk(NSError* error, std::string* out);
id<MTLDevice> MetalCreateDevice(std::string* note);
id<MTLLibrary> LoadLibrary(id<MTLDevice> device, std::string* error);

struct ThreadgroupSize {
  int x;
  int y;
};
ThreadgroupSize ChooseThreadgroupSize(id<MTLComputePipelineState> pipeline, int user_x, int user_y);
void Dispatch2D(id<MTLComputeCommandEncoder> encoder, int width, int height, ThreadgroupSize tg);
void Dispatch1D(id<MTLComputeCommandEncoder> encoder, int count, int threads_per_group);

// Vector operations
class MetalVectorOps {
public:
    MetalVectorOps(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library);
    bool isReady(std::string* error);

    bool dot(id<MTLBuffer> a_buf, id<MTLBuffer> b_buf, uint count, float* out);
    bool norm2(id<MTLBuffer> v_buf, uint count, float* out);
    bool set(id<MTLBuffer> v, float value, uint count);
    bool copy(id<MTLBuffer> src, id<MTLBuffer> dst, uint count);
    bool axpy(id<MTLBuffer> y_buf, id<MTLBuffer> x_buf, float a, uint count);
    bool scale(id<MTLBuffer> v_buf, float a_scale, uint count);
    bool lincomb2(id<MTLBuffer> out_buf, id<MTLBuffer> a_buf, id<MTLBuffer> b_buf, float alpha, float beta, uint count);

private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _dot_pipeline;
    id<MTLComputePipelineState> _sum_pipeline;
    id<MTLComputePipelineState> _set_pipeline;
    id<MTLComputePipelineState> _copy_pipeline;
    id<MTLComputePipelineState> _axpy_pipeline;
    id<MTLComputePipelineState> _scale_pipeline;
    id<MTLComputePipelineState> _lincomb2_pipeline;

    id<MTLBuffer> _reduce_a;
    id<MTLBuffer> _reduce_b;
    id<MTLBuffer> _reduce_result;
};


#endif // METAL_UTILS_H
