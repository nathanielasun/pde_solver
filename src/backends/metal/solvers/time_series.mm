#import "time_series.h"
#import "../utils/metal_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "coefficient_evaluator.h"
#include "expression_eval.h"

namespace {

DeviceExpr ToDeviceExpr(const BoundaryCondition::Expression& expr) {
  return {static_cast<float>(expr.constant), static_cast<float>(expr.x), static_cast<float>(expr.y)};
}

DeviceBC ToDeviceBC(const BoundaryCondition& bc) {
  return {static_cast<int>(bc.kind), ToDeviceExpr(bc.value), ToDeviceExpr(bc.alpha), ToDeviceExpr(bc.beta), ToDeviceExpr(bc.gamma)};
}

// Buffered frame data for batch I/O
struct FrameBuffer {
    std::vector<float> grid;
    std::vector<float> velocity;  // Only used for u_tt problems
    double time;
    int frame_index;
};

}  // namespace

SolveOutput SolvePDETimeSeriesMetal(const SolveInput& input,
                                     const FrameCallback& on_frame,
                                     const ProgressCallback& progress) {
    const Domain& d = input.domain;

    if (d.nx < 3 || d.ny < 3) {
        return {"grid must be at least 3x3", {}};
    }
    if (d.nz > 1) {
        return {"Metal time-series backend supports 2D domains only", {}};
    }
    if (d.xmax <= d.xmin || d.ymax <= d.ymin) {
        return {"domain bounds are invalid", {}};
    }
    if (!input.time.enabled) {
        return {"time-series solver requires time settings", {}};
    }

    const bool has_ut = std::abs(input.pde.ut) > 1e-12;
    const bool has_utt = std::abs(input.pde.utt) > 1e-12;
    if (!has_ut && !has_utt) {
        return {"time-series solver requires ut or utt term", {}};
    }

    const int nx = d.nx, ny = d.ny, total = nx * ny;
    const float dx = static_cast<float>((d.xmax - d.xmin) / (nx - 1));
    const float dy = static_cast<float>((d.ymax - d.ymin) / (ny - 1));

    const int frames = input.time.frames;
    const double t_start = input.time.t_start;
    const double dt_d = input.time.dt;
    const float dt = static_cast<float>(dt_d);

    // Calculate buffer capacity based on memory limit
    const size_t frame_bytes = static_cast<size_t>(total) * sizeof(float) * (has_utt ? 2 : 1);
    const size_t buffer_bytes = static_cast<size_t>(input.time.buffer_mb) * 1024 * 1024;
    const int max_buffered_frames = std::max(1, static_cast<int>(buffer_bytes / frame_bytes));
    const int buffer_capacity = std::min(max_buffered_frames, frames);

    NSLog(@"Metal time-series: buffer_mb=%d, frame_size=%.2fMB, buffer_capacity=%d frames",
          input.time.buffer_mb, static_cast<double>(frame_bytes) / (1024 * 1024), buffer_capacity);

    // Compute PDE coefficients
    CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
    if (!coeff_eval.ok) {
        return {coeff_eval.error, {}};
    }
    if (coeff_eval.has_variable) {
        return {"Metal time-series does not support variable coefficients yet", {}};
    }

    const float ax = static_cast<float>(input.pde.a / (dx * dx));
    const float by = static_cast<float>(input.pde.b / (dy * dy));
    const float cx = static_cast<float>(input.pde.c / (2.0 * dx));
    const float dyc = static_cast<float>(input.pde.d / (2.0 * dy));
    const float f_const = static_cast<float>(input.pde.f);
    const float ut_coeff = static_cast<float>(input.pde.ut);
    const float utt_coeff = static_cast<float>(input.pde.utt);

    // Create Metal resources
    std::string device_note;
    id<MTLDevice> device = MetalCreateDevice(&device_note);
    if (!device) {
        const std::string msg = device_note.empty()
            ? "Metal device unavailable"
            : "Metal device unavailable: " + device_note;
        return {msg, {}};
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return {"Failed to create Metal command queue", {}};

    std::string library_error;
    id<MTLLibrary> library = LoadLibrary(device, &library_error);
    if (!library) return {"Failed to load metal library: " + library_error, {}};

    NSLog(@"Metal time-series: Library loaded with %lu functions", (unsigned long)[[library functionNames] count]);

    // Load kernel functions
    NSString* kernel_name = has_utt ? @"time_step_second_order" : @"time_step_first_order";
    id<MTLFunction> time_step_fn = [library newFunctionWithName:kernel_name];
    id<MTLFunction> boundary_fn = [library newFunctionWithName:@"apply_boundaries"];
    if (!time_step_fn) {
        NSLog(@"Metal time-series: Available functions: %@", [library functionNames]);
        return {"Failed to load time-stepping kernel: " + std::string([kernel_name UTF8String]), {}};
    }
    if (!boundary_fn) {
        return {"Failed to load boundary kernel", {}};
    }
    NSLog(@"Metal time-series: Loaded kernel '%@' and boundary kernel", kernel_name);

    NSError* ns_error = nil;
    id<MTLComputePipelineState> time_step_pipeline = [device newComputePipelineStateWithFunction:time_step_fn error:&ns_error];
    id<MTLComputePipelineState> boundary_pipeline = [device newComputePipelineStateWithFunction:boundary_fn error:&ns_error];
    if (!time_step_pipeline || !boundary_pipeline) {
        return {"Failed to create Metal pipeline state", {}};
    }

    // Domain and boundary info
    DomainInfo domain_info = {nx, ny, static_cast<float>(d.xmin), static_cast<float>(d.xmax),
                              static_cast<float>(d.ymin), static_cast<float>(d.ymax), dx, dy};
    DeviceBC left = ToDeviceBC(input.bc.left);
    DeviceBC right = ToDeviceBC(input.bc.right);
    DeviceBC bottom = ToDeviceBC(input.bc.bottom);
    DeviceBC top = ToDeviceBC(input.bc.top);

    // Allocate GPU buffers for computation
    id<MTLBuffer> grid_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> next_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> velocity_buf = nil;
    if (has_utt) {
        velocity_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    }

    id<MTLBuffer> domain_b = [device newBufferWithBytes:&domain_info length:sizeof(DomainInfo) options:MTLResourceStorageModeShared];
    id<MTLBuffer> left_b = [device newBufferWithBytes:&left length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> right_b = [device newBufferWithBytes:&right length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bottom_b = [device newBufferWithBytes:&bottom length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> top_b = [device newBufferWithBytes:&top length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];

    // Shared buffer for GPU→CPU transfers
    id<MTLBuffer> host_grid_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModeShared];
    id<MTLBuffer> host_velocity_buf = nil;
    if (has_utt) {
        host_velocity_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModeShared];
    }

    if (!grid_buf || !next_buf || !domain_b || !left_b || !right_b || !bottom_b || !top_b || !host_grid_buf) {
        return {"Failed to allocate Metal buffers", {}};
    }
    if (has_utt && (!velocity_buf || !host_velocity_buf)) {
        return {"Failed to allocate velocity buffers", {}};
    }

    // Allocate frame ring buffer for batched I/O
    // We store frames in CPU memory, then write them all at once
    std::vector<FrameBuffer> frame_ring(buffer_capacity);
    for (int i = 0; i < buffer_capacity; ++i) {
        frame_ring[i].grid.resize(total);
        if (has_utt) {
            frame_ring[i].velocity.resize(total);
        }
    }

    // Initialize grid with initial conditions or zeros
    std::vector<float> initial_grid(total, 0.0f);
    std::vector<float> initial_velocity(total, 0.0f);
    if (!input.initial_grid.empty()) {
        if (input.initial_grid.size() != static_cast<size_t>(total)) {
            return {"initial grid size mismatch", {}};
        }
        for (int i = 0; i < total; ++i) {
            initial_grid[i] = static_cast<float>(input.initial_grid[i]);
        }
    }
    if (has_utt && !input.initial_velocity.empty()) {
        if (input.initial_velocity.size() != static_cast<size_t>(total)) {
            return {"initial velocity size mismatch", {}};
        }
        for (int i = 0; i < total; ++i) {
            initial_velocity[i] = static_cast<float>(input.initial_velocity[i]);
        }
    } else if (has_utt && !input.initial_grid.empty() && input.initial_velocity.empty()) {
        return {"checkpoint restart for u_tt requires velocity data", {}};
    }

    // Upload initial data to GPU
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return {"Failed to create command buffer for initialization", {}};
        }
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) {
            return {"Failed to create blit encoder for initialization", {}};
        }

        memcpy([host_grid_buf contents], initial_grid.data(), sizeof(float) * total);
        [blit copyFromBuffer:host_grid_buf sourceOffset:0 toBuffer:grid_buf destinationOffset:0 size:sizeof(float)*total];

        if (has_utt) {
            memcpy([host_velocity_buf contents], initial_velocity.data(), sizeof(float) * total);
            [blit copyFromBuffer:host_velocity_buf sourceOffset:0 toBuffer:velocity_buf destinationOffset:0 size:sizeof(float)*total];
        }

        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Apply initial boundary conditions
    ThreadgroupSize tg = ChooseThreadgroupSize(boundary_pipeline, input.solver.metal_tg_x, input.solver.metal_tg_y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return {"Failed to create command buffer for boundary", {}};
        }
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        if (!encoder) {
            return {"Failed to create compute encoder for boundary", {}};
        }
        @try {
            [encoder setComputePipelineState:boundary_pipeline];
            [encoder setBuffer:grid_buf offset:0 atIndex:0];
            [encoder setBuffer:domain_b offset:0 atIndex:1];
            [encoder setBuffer:left_b offset:0 atIndex:2];
            [encoder setBuffer:right_b offset:0 atIndex:3];
            [encoder setBuffer:bottom_b offset:0 atIndex:4];
            [encoder setBuffer:top_b offset:0 atIndex:5];
            Dispatch2D(encoder, nx, ny, tg);
        } @finally {
            [encoder endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Report progress
    if (progress) {
        progress("solve_total", static_cast<double>(frames));
        progress("time", 0.0);
    }

    ThreadgroupSize step_tg = ChooseThreadgroupSize(time_step_pipeline, input.solver.metal_tg_x, input.solver.metal_tg_y);
    NSLog(@"Metal time-series: Dispatching time step kernel, grid=%dx%d, tg=%dx%d", nx, ny, step_tg.x, step_tg.y);

    // Timing accumulators for profiling
    double time_gpu_compute = 0, time_gpu_copy = 0, time_convert = 0, time_callback = 0;
    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto elapsed_ms = [](auto start) {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    };

    // Final result grid
    std::vector<double> result_grid(total);
    std::vector<double> result_velocity;
    if (has_utt) {
        result_velocity.resize(total);
    }

    // Process frames in batches
    int frame = 0;
    while (frame < frames) {
        if (input.cancel && input.cancel->load()) {
            return {"solve cancelled", {}};
        }

        // Determine batch size
        const int batch_size = std::min(buffer_capacity, frames - frame);
        int frames_computed = 0;

        // =====================================================
        // PHASE 1: GPU Compute (batch all time steps, minimal sync)
        // =====================================================
        auto t_compute_start = now();

        // Use a single command buffer for the entire batch to reduce overhead
        @autoreleasepool {
            for (int b = 0; b < batch_size; ++b) {
                const int current_frame = frame + b;
                const double t = t_start + current_frame * dt_d;

                // Record frame metadata
                frame_ring[b].time = t;
                frame_ring[b].frame_index = current_frame;
                frames_computed = b + 1;

                // Copy current state to CPU buffer (we need this for output)
                // Use a single command buffer for efficiency
                id<MTLCommandBuffer> copy_cmd = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [copy_cmd blitCommandEncoder];
                [blit copyFromBuffer:grid_buf sourceOffset:0 toBuffer:host_grid_buf destinationOffset:0 size:sizeof(float)*total];
                if (has_utt) {
                    [blit copyFromBuffer:velocity_buf sourceOffset:0 toBuffer:host_velocity_buf destinationOffset:0 size:sizeof(float)*total];
                }
                [blit endEncoding];
                [copy_cmd commit];
                [copy_cmd waitUntilCompleted];

                // Copy to ring buffer (CPU memory)
                memcpy(frame_ring[b].grid.data(), [host_grid_buf contents], sizeof(float) * total);
                if (has_utt) {
                    memcpy(frame_ring[b].velocity.data(), [host_velocity_buf contents], sizeof(float) * total);
                }

                // Don't step if this is the last frame overall
                if (current_frame + 1 >= frames) {
                    break;
                }

                // Batch GPU work: time step + boundary in a single command buffer
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                if (!cmd) {
                    return {"Failed to create command buffer", {}};
                }

                // Time step kernel
                id<MTLComputeCommandEncoder> step_encoder = [cmd computeCommandEncoder];
                if (!step_encoder) {
                    return {"Failed to create compute encoder", {}};
                }
                [step_encoder setComputePipelineState:time_step_pipeline];
                [step_encoder setBuffer:grid_buf offset:0 atIndex:0];
                [step_encoder setBuffer:next_buf offset:0 atIndex:1];

                if (has_utt) {
                    [step_encoder setBuffer:velocity_buf offset:0 atIndex:2];
                    [step_encoder setBuffer:domain_b offset:0 atIndex:3];
                    [step_encoder setBuffer:left_b offset:0 atIndex:4];
                    [step_encoder setBuffer:right_b offset:0 atIndex:5];
                    [step_encoder setBuffer:bottom_b offset:0 atIndex:6];
                    [step_encoder setBuffer:top_b offset:0 atIndex:7];
                    [step_encoder setBytes:&ax length:sizeof(float) atIndex:8];
                    [step_encoder setBytes:&by length:sizeof(float) atIndex:9];
                    [step_encoder setBytes:&cx length:sizeof(float) atIndex:10];
                    [step_encoder setBytes:&dyc length:sizeof(float) atIndex:11];
                    [step_encoder setBytes:&f_const length:sizeof(float) atIndex:12];
                    [step_encoder setBytes:&ut_coeff length:sizeof(float) atIndex:13];
                    [step_encoder setBytes:&utt_coeff length:sizeof(float) atIndex:14];
                    [step_encoder setBytes:&dt length:sizeof(float) atIndex:15];
                } else {
                    [step_encoder setBuffer:domain_b offset:0 atIndex:2];
                    [step_encoder setBuffer:left_b offset:0 atIndex:3];
                    [step_encoder setBuffer:right_b offset:0 atIndex:4];
                    [step_encoder setBuffer:bottom_b offset:0 atIndex:5];
                    [step_encoder setBuffer:top_b offset:0 atIndex:6];
                    [step_encoder setBytes:&ax length:sizeof(float) atIndex:7];
                    [step_encoder setBytes:&by length:sizeof(float) atIndex:8];
                    [step_encoder setBytes:&cx length:sizeof(float) atIndex:9];
                    [step_encoder setBytes:&dyc length:sizeof(float) atIndex:10];
                    [step_encoder setBytes:&f_const length:sizeof(float) atIndex:11];
                    [step_encoder setBytes:&ut_coeff length:sizeof(float) atIndex:12];
                    [step_encoder setBytes:&dt length:sizeof(float) atIndex:13];
                }
                Dispatch2D(step_encoder, nx, ny, step_tg);
                [step_encoder endEncoding];

                // Swap buffers (pointer swap, no GPU work)
                std::swap(grid_buf, next_buf);

                // Boundary conditions kernel (in same command buffer timeline)
                id<MTLComputeCommandEncoder> bc_encoder = [cmd computeCommandEncoder];
                [bc_encoder setComputePipelineState:boundary_pipeline];
                [bc_encoder setBuffer:grid_buf offset:0 atIndex:0];
                [bc_encoder setBuffer:domain_b offset:0 atIndex:1];
                [bc_encoder setBuffer:left_b offset:0 atIndex:2];
                [bc_encoder setBuffer:right_b offset:0 atIndex:3];
                [bc_encoder setBuffer:bottom_b offset:0 atIndex:4];
                [bc_encoder setBuffer:top_b offset:0 atIndex:5];
                Dispatch2D(bc_encoder, nx, ny, tg);
                [bc_encoder endEncoding];

                // Commit and wait (required for correctness since we read back each frame)
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }

        time_gpu_compute += elapsed_ms(t_compute_start);

        // =====================================================
        // PHASE 2: Batch I/O (convert and write all buffered frames)
        // =====================================================
        for (int b = 0; b < frames_computed; ++b) {
            const FrameBuffer& fb = frame_ring[b];

            // Report progress
            if (progress) {
                const double frac = (frames <= 1) ? 1.0 : static_cast<double>(fb.frame_index) / static_cast<double>(frames - 1);
                progress("time", frac);
                progress("iteration", static_cast<double>(fb.frame_index + 1));
            }

            // Convert float to double
            auto t_convert = now();
            for (int i = 0; i < total; ++i) {
                result_grid[i] = static_cast<double>(fb.grid[i]);
            }
            if (has_utt) {
                for (int i = 0; i < total; ++i) {
                    result_velocity[i] = static_cast<double>(fb.velocity[i]);
                }
            }
            time_convert += elapsed_ms(t_convert);

            // Call frame callback (writes VTK)
            auto t_callback = now();
            if (on_frame) {
                const std::vector<double>* velocity_ptr = has_utt ? &result_velocity : nullptr;
                if (!on_frame(fb.frame_index, fb.time, result_grid, velocity_ptr)) {
                    // Callback requested stop
                    NSLog(@"Metal time-series timing (ms): GPU-compute=%.1f, convert=%.1f, callback=%.1f",
                          time_gpu_compute, time_convert, time_callback);
                    SolveOutput out;
                    out.grid = std::move(result_grid);
                    return out;
                }
            }
            time_callback += elapsed_ms(t_callback);
        }

        frame += frames_computed;
    }

    // Print timing summary
    NSLog(@"Metal time-series timing (ms): GPU-compute=%.1f, convert=%.1f, callback=%.1f, total=%.1f",
          time_gpu_compute, time_convert, time_callback,
          time_gpu_compute + time_convert + time_callback);
    NSLog(@"Metal time-series: GPU compute is %.1f%% of total time",
          100.0 * time_gpu_compute / (time_gpu_compute + time_convert + time_callback + 0.001));

    SolveOutput out;
    out.grid = std::move(result_grid);
    return out;
}
