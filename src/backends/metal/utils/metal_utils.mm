#import "metal_utils.h"
#import <TargetConditionals.h>

#include <algorithm>

namespace {

int ReduceGroupCount(int count) {
  const int groups = (count + kReduceGroupSize - 1) / kReduceGroupSize;
  return std::max(1, groups);
}

void DispatchReduce(id<MTLComputeCommandEncoder> encoder, int groups) {
  MTLSize threadsPerThreadgroup = MTLSizeMake(kReduceGroupSize, 1, 1);
  MTLSize threadgroups = MTLSizeMake(groups, 1, 1);
  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
}

} // namespace


bool MetalOk(NSError* error, std::string* out) {
  if (!error) return true;
  if (out) *out = [[error localizedDescription] UTF8String];
  return false;
}

id<MTLDevice> MetalCreateDevice(std::string* note) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
      if (note) *note = [[device name] UTF8String];
      return device;
    }
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices && [devices count] > 0) {
      id<MTLDevice> fallback = devices[0];
      if (note) *note = [[fallback name] UTF8String];
      return fallback;
    }
#endif
    if (note) *note = "no Metal device available";
    return nil;
  }
}

id<MTLLibrary> LoadLibrary(id<MTLDevice> device, std::string* error) {
  NSString* resource = @"kernels"; // New consolidated name
  NSBundle* bundle = [NSBundle mainBundle];
  NSError* ns_error = nil;

  // Try loading from a pre-compiled .metallib
  NSString* metallib_path = [bundle pathForResource:resource ofType:@"metallib"];
  if (!metallib_path) {
    // Fallback for command-line tools or different project structures
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    metallib_path = [cwd stringByAppendingPathComponent:@"kernels.metallib"];
  }
   if (!metallib_path) { // Search relative to executable
      metallib_path = [[bundle resourcePath] stringByAppendingPathComponent:@"kernels.metallib"];
  }

  if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path]) {
    NSURL* url = [NSURL fileURLWithPath:metallib_path];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&ns_error];
    if (lib) return lib;
  }

  // Fallback to compiling from .metal source
  NSString* source_path = [bundle pathForResource:resource ofType:@"metal"];
  if (!source_path) {
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    // Look in the new solvers directory
    source_path = [cwd stringByAppendingPathComponent:@"solvers/kernels.metal"];
  }

  NSString* source = [NSString stringWithContentsOfFile:source_path encoding:NSUTF8StringEncoding error:&ns_error];
  if (!source) {
    if (error) *error = ns_error ? [[ns_error localizedDescription] UTF8String] : "failed to load kernels.metal";
    return nil;
  }

  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&ns_error];
  if (!lib && error) {
    *error = ns_error ? [[ns_error localizedDescription] UTF8String] : "failed to compile metal source";
  }
  return lib;
}


ThreadgroupSize ChooseThreadgroupSize(id<MTLComputePipelineState> pipeline, int user_x, int user_y) {
  const int max_threads = static_cast<int>(pipeline.maxTotalThreadsPerThreadgroup);
  int tg_x = user_x > 0 ? user_x : static_cast<int>(pipeline.threadExecutionWidth);
  int tg_y = user_y > 0 ? user_y : std::max(1, max_threads / tg_x);
  
  if (tg_x * tg_y > max_threads) {
      tg_y = std::max(1, max_threads / tg_x);
  }
  return {tg_x, tg_y};
}

void Dispatch2D(id<MTLComputeCommandEncoder> encoder, int width, int height, ThreadgroupSize tg) {
  MTLSize threadsPerThreadgroup = MTLSizeMake(tg.x, tg.y, 1);
  MTLSize threadgroups = MTLSizeMake((width + tg.x - 1) / tg.x, (height + tg.y - 1) / tg.y, 1);
  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
}

void Dispatch1D(id<MTLComputeCommandEncoder> encoder, int count, int threads_per_group) {
  MTLSize threadsPerThreadgroup = MTLSizeMake(threads_per_group, 1, 1);
  MTLSize threadgroups = MTLSizeMake((count + threads_per_group - 1) / threads_per_group, 1, 1);
  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
}


// MetalVectorOps implementation
MetalVectorOps::MetalVectorOps(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library)
: _device(device), _queue(queue) {
    NSError* ns_error = nil;
    #define CREATE_PIPELINE(name, var) \
        id<MTLFunction> fn_##name = [library newFunctionWithName:@#name]; \
        if (fn_##name) var = [_device newComputePipelineStateWithFunction:fn_##name error:&ns_error];

    CREATE_PIPELINE(reduce_dot, _dot_pipeline);
    CREATE_PIPELINE(reduce_sum, _sum_pipeline);
    CREATE_PIPELINE(vec_set, _set_pipeline);
    CREATE_PIPELINE(vec_copy, _copy_pipeline);
    CREATE_PIPELINE(vec_axpy, _axpy_pipeline);
    CREATE_PIPELINE(vec_scale, _scale_pipeline);
    CREATE_PIPELINE(vec_lincomb2, _lincomb2_pipeline);
    
    #undef CREATE_PIPELINE

    // Allocate reduction buffers once
    const int reduce_capacity = 1024; // A reasonably large capacity
    _reduce_a = [_device newBufferWithLength:sizeof(float) * reduce_capacity options:MTLResourceStorageModePrivate];
    _reduce_b = [_device newBufferWithLength:sizeof(float) * reduce_capacity options:MTLResourceStorageModePrivate];
    _reduce_result = [_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
}

bool MetalVectorOps::isReady(std::string* error) {
    if (!_dot_pipeline || !_sum_pipeline || !_set_pipeline || !_copy_pipeline || !_axpy_pipeline || !_scale_pipeline || !_lincomb2_pipeline) {
        if(error) *error = "One or more vector ops pipelines failed to create.";
        return false;
    }
    if (!_reduce_a || !_reduce_b || !_reduce_result) {
        if(error) *error = "Failed to allocate reduction buffers.";
        return false;
    }
    return true;
}


bool MetalVectorOps::dot(id<MTLBuffer> a_buf, id<MTLBuffer> b_buf, uint count, float* out) {
    if (!out) return false;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        if (!encoder) return false;

        uint reduce_count = count;
        id<MTLBuffer> reduce_in = nil;
        id<MTLBuffer> reduce_out = _reduce_a;

        uint groups = ReduceGroupCount(reduce_count);
        if (groups > _reduce_a.length / sizeof(float)) {
            [encoder endEncoding];  // Must end encoder before returning
            return false;
        }

        [encoder setComputePipelineState:_dot_pipeline];
        [encoder setBuffer:a_buf offset:0 atIndex:0];
        [encoder setBuffer:b_buf offset:0 atIndex:1];
        [encoder setBuffer:reduce_out offset:0 atIndex:2];
        [encoder setBytes:&reduce_count length:sizeof(uint) atIndex:3];
        DispatchReduce(encoder, groups);
        
        reduce_count = groups;
        reduce_in = reduce_out;
        reduce_out = _reduce_b;

        while (reduce_count > 1) {
            groups = ReduceGroupCount(reduce_count);
            [encoder setComputePipelineState:_sum_pipeline];
            [encoder setBuffer:reduce_in offset:0 atIndex:0];
            [encoder setBuffer:reduce_out offset:0 atIndex:1];
            [encoder setBytes:&reduce_count length:sizeof(uint) atIndex:2];
            DispatchReduce(encoder, groups);
            reduce_count = groups;
            if (reduce_count == 1) {
                reduce_in = reduce_out;
                break;
            }
            std::swap(reduce_in, reduce_out);
        }

        [encoder endEncoding];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:reduce_in sourceOffset:0 toBuffer:_reduce_result destinationOffset:0 size:sizeof(float)];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        *out = *(static_cast<const float*>([_reduce_result contents]));
        return true;
    }
}

bool MetalVectorOps::norm2(id<MTLBuffer> v_buf, uint count, float* out) {
    if (!out) return false;
    float dot_val = 0.0f;
    if (!dot(v_buf, v_buf, count, &dot_val)) return false;
    *out = std::sqrt(std::max(0.0f, dot_val));
    return true;
}

#define DISPATCH_VEC_OP(pipeline, ...) \
    @autoreleasepool { \
        id<MTLCommandBuffer> cmd = [_queue commandBuffer]; \
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder]; \
        [encoder setComputePipelineState:pipeline]; \
        __VA_ARGS__ \
        Dispatch1D(encoder, count, kReduceGroupSize); \
        [encoder endEncoding]; \
        [cmd commit]; \
        [cmd waitUntilCompleted]; \
        return true; \
    }

bool MetalVectorOps::set(id<MTLBuffer> v, float value, uint count) {
    DISPATCH_VEC_OP(_set_pipeline, \
        [encoder setBuffer:v offset:0 atIndex:0]; \
        [encoder setBytes:&value length:sizeof(float) atIndex:1]; \
        [encoder setBytes:&count length:sizeof(uint) atIndex:2]; \
    );
}

bool MetalVectorOps::copy(id<MTLBuffer> src, id<MTLBuffer> dst, uint count) {
    DISPATCH_VEC_OP(_copy_pipeline, \
        [encoder setBuffer:src offset:0 atIndex:0]; \
        [encoder setBuffer:dst offset:0 atIndex:1]; \
        [encoder setBytes:&count length:sizeof(uint) atIndex:2]; \
    );
}

bool MetalVectorOps::axpy(id<MTLBuffer> y_buf, id<MTLBuffer> x_buf, float a, uint count) {
    DISPATCH_VEC_OP(_axpy_pipeline, \
        [encoder setBuffer:y_buf offset:0 atIndex:0]; \
        [encoder setBuffer:x_buf offset:0 atIndex:1]; \
        [encoder setBytes:&a length:sizeof(float) atIndex:2]; \
        [encoder setBytes:&count length:sizeof(uint) atIndex:3]; \
    );
}

bool MetalVectorOps::scale(id<MTLBuffer> v_buf, float a_scale, uint count) {
    DISPATCH_VEC_OP(_scale_pipeline, \
        [encoder setBuffer:v_buf offset:0 atIndex:0]; \
        [encoder setBytes:&a_scale length:sizeof(float) atIndex:1]; \
        [encoder setBytes:&count length:sizeof(uint) atIndex:2]; \
    );
}

bool MetalVectorOps::lincomb2(id<MTLBuffer> out_buf, id<MTLBuffer> a_buf, id<MTLBuffer> b_buf, float alpha, float beta, uint count) {
    DISPATCH_VEC_OP(_lincomb2_pipeline, \
        [encoder setBuffer:out_buf offset:0 atIndex:0]; \
        [encoder setBuffer:a_buf offset:0 atIndex:1]; \
        [encoder setBuffer:b_buf offset:0 atIndex:2]; \
        [encoder setBytes:&alpha length:sizeof(float) atIndex:3]; \
        [encoder setBytes:&beta length:sizeof(float) atIndex:4]; \
        [encoder setBytes:&count length:sizeof(uint) atIndex:5]; \
    );
}

#undef DISPATCH_VEC_OP
