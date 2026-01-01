#import "relaxation.h"
#import "../utils/metal_utils.h"

#include <algorithm>
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

}

SolveOutput MetalSolveRelaxation(const SolveInput& input) {
    const Domain& d = input.domain;
    std::string error;

    const int nx = d.nx, ny = d.ny, total = nx * ny;
    const float dx = (d.xmax - d.xmin) / (nx - 1);
    const float dy = (d.ymax - d.ymin) / (ny - 1);

    CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
    if (!coeff_eval.ok) {
        return {coeff_eval.error, {}};
    }
    const bool has_var_coeff = coeff_eval.has_variable;

    const bool has_rhs_expr = !input.pde.rhs_latex.empty();
    std::optional<ExpressionEvaluator> rhs_eval;
    if (has_rhs_expr) {
        ExpressionEvaluator eval = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
        if (!eval.ok()) {
            return {"invalid RHS expression: " + eval.error(), {}};
        }
        rhs_eval.emplace(std::move(eval));
    }

    const float ax_const = input.pde.a / (dx * dx);
    const float by_const = input.pde.b / (dy * dy);
    const float cx_const = input.pde.c / (2.0f * dx);
    const float dyc_const = input.pde.d / (2.0f * dy);
    const float center_const = -2.0f * ax_const - 2.0f * by_const + input.pde.e;
    const float f_const = input.pde.f;
    const bool use_coeff_buffers = has_var_coeff || has_rhs_expr;

    std::vector<float> rhs_values;
    std::vector<float> ax_values;
    std::vector<float> by_values;
    std::vector<float> cx_values;
    std::vector<float> dyc_values;
    std::vector<float> center_values;
    if (use_coeff_buffers) {
        rhs_values.assign(total, f_const);
        ax_values.assign(total, ax_const);
        by_values.assign(total, by_const);
        cx_values.assign(total, cx_const);
        dyc_values.assign(total, dyc_const);
        center_values.assign(total, center_const);
        for (int j = 0; j < ny; ++j) {
            const double y = d.ymin + j * dy;
            for (int i = 0; i < nx; ++i) {
                const double x = d.xmin + i * dx;
                const int idx = j * nx + i;
                if (rhs_eval) {
                    rhs_values[idx] = static_cast<float>(rhs_eval->Eval(x, y, 0.0, 0.0));
                }
                const double a_val = EvalCoefficient(coeff_eval.a, input.pde.a, x, y, 0.0, 0.0);
                const double b_val = EvalCoefficient(coeff_eval.b, input.pde.b, x, y, 0.0, 0.0);
                const double c_val = EvalCoefficient(coeff_eval.c, input.pde.c, x, y, 0.0, 0.0);
                const double d_val = EvalCoefficient(coeff_eval.d, input.pde.d, x, y, 0.0, 0.0);
                const double e_val = EvalCoefficient(coeff_eval.e, input.pde.e, x, y, 0.0, 0.0);
                const double ax = a_val / (dx * dx);
                const double by = b_val / (dy * dy);
                const double cx = c_val / (2.0 * dx);
                const double dyc = d_val / (2.0 * dy);
                const double center = -2.0 * ax - 2.0 * by + e_val;
                if (std::abs(center) < 1e-6) {
                    return {"Degenerate PDE center coefficient", {}};
                }
                ax_values[idx] = static_cast<float>(ax);
                by_values[idx] = static_cast<float>(by);
                cx_values[idx] = static_cast<float>(cx);
                dyc_values[idx] = static_cast<float>(dyc);
                center_values[idx] = static_cast<float>(center);
            }
        }
    } else {
        if (std::abs(center_const) < 1e-6f) {
            return {"Degenerate PDE center coefficient", {}};
        }
    }

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
    id<MTLLibrary> library = LoadLibrary(device, &error);
    if (!library) return {"Failed to load metal library: " + error, {}};

    NSError* ns_error = nil;
    id<MTLFunction> boundary_fn = [library newFunctionWithName:@"apply_boundaries"];
    NSString* jacobi_name = use_coeff_buffers ? @"jacobi_step_coeff" : @"jacobi_step";
    NSString* rbgs_name = use_coeff_buffers ? @"rbgs_step_coeff" : @"rbgs_step";
    id<MTLFunction> jacobi_fn = [library newFunctionWithName:jacobi_name];
    id<MTLFunction> rbgs_fn = [library newFunctionWithName:rbgs_name];
    id<MTLFunction> reduce_fn = [library newFunctionWithName:@"reduce_max"];
    if (!boundary_fn || !jacobi_fn || !rbgs_fn || !reduce_fn) return {"A required Metal kernel function was not found.", {}};

    id<MTLComputePipelineState> boundary_pipeline = [device newComputePipelineStateWithFunction:boundary_fn error:&ns_error];
    id<MTLComputePipelineState> jacobi_pipeline = [device newComputePipelineStateWithFunction:jacobi_fn error:&ns_error];
    id<MTLComputePipelineState> rbgs_pipeline = [device newComputePipelineStateWithFunction:rbgs_fn error:&ns_error];
    id<MTLComputePipelineState> reduce_pipeline = [device newComputePipelineStateWithFunction:reduce_fn error:&ns_error];
    if (!boundary_pipeline || !jacobi_pipeline || !rbgs_pipeline || !reduce_pipeline) return {"Failed to create Metal pipeline state.", {}};

    DomainInfo domain_info = {nx, ny, (float)d.xmin, (float)d.xmax, (float)d.ymin, (float)d.ymax, dx, dy};
    DeviceBC left = ToDeviceBC(input.bc.left);
    DeviceBC right = ToDeviceBC(input.bc.right);
    DeviceBC bottom = ToDeviceBC(input.bc.bottom);
    DeviceBC top = ToDeviceBC(input.bc.top);

    id<MTLBuffer> grid_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> next_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> delta_buf = [device newBufferWithLength:sizeof(float)*total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> domain_b = [device newBufferWithBytes:&domain_info length:sizeof(DomainInfo) options:MTLResourceStorageModeShared];
    id<MTLBuffer> left_b = [device newBufferWithBytes:&left length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> right_b = [device newBufferWithBytes:&right length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bottom_b = [device newBufferWithBytes:&bottom length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> top_b = [device newBufferWithBytes:&top length:sizeof(DeviceBC) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ax_b = nil;
    id<MTLBuffer> by_b = nil;
    id<MTLBuffer> cx_b = nil;
    id<MTLBuffer> dyc_b = nil;
    id<MTLBuffer> center_b = nil;
    id<MTLBuffer> rhs_b = nil;
    if (use_coeff_buffers) {
        ax_b = [device newBufferWithBytes:ax_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        by_b = [device newBufferWithBytes:by_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        cx_b = [device newBufferWithBytes:cx_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        dyc_b = [device newBufferWithBytes:dyc_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        center_b = [device newBufferWithBytes:center_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        rhs_b = [device newBufferWithBytes:rhs_values.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
        if (!ax_b || !by_b || !cx_b || !dyc_b || !center_b || !rhs_b) {
            return {"Failed to allocate Metal coefficient buffers", {}};
        }
    }
    
    // Reduction buffers sized for first-pass block reductions.
    const uint32_t reduce_capacity = std::max(1u, (static_cast<uint32_t>(total) + kReduceGroupSize - 1) / kReduceGroupSize);
    id<MTLBuffer> reduce_a = [device newBufferWithLength:sizeof(float) * reduce_capacity options:MTLResourceStorageModePrivate];
    id<MTLBuffer> reduce_res = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> host_res_buf = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModeShared];

    @autoreleasepool {
        id<MTLCommandBuffer> init_cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [init_cmd blitCommandEncoder];
        [blit fillBuffer:grid_buf range:NSMakeRange(0, sizeof(float) * total) value:0];
        [blit endEncoding];  // Must end encoder before committing
        [init_cmd commit];
        [init_cmd waitUntilCompleted];
    }
    
    ThreadgroupSize tg = ChooseThreadgroupSize(jacobi_pipeline, input.solver.metal_tg_x, input.solver.metal_tg_y);
    const int reduce_interval = std::max(1, input.solver.metal_reduce_interval);

    for (int iter = 0; iter < input.solver.max_iter; ) {
        int batch = std::min(reduce_interval, input.solver.max_iter - iter);
        if (batch <= 0) break;
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            if (!encoder) {
                return {"Failed to create Metal compute encoder", {}};
            }

            // Declare reduce_in outside try block so it's accessible after
            id<MTLBuffer> reduce_in = delta_buf;

            @try {
            for (int step = 0; step < batch; ++step) {
                if (input.solver.method == SolveMethod::Jacobi) {
                    [encoder setComputePipelineState:jacobi_pipeline];
                    [encoder setBuffer:grid_buf offset:0 atIndex:0];
                    [encoder setBuffer:next_buf offset:0 atIndex:1];
                    [encoder setBuffer:delta_buf offset:0 atIndex:2];
                    [encoder setBuffer:domain_b offset:0 atIndex:3];
                    [encoder setBuffer:left_b offset:0 atIndex:4];
                    [encoder setBuffer:right_b offset:0 atIndex:5];
                    [encoder setBuffer:bottom_b offset:0 atIndex:6];
                    [encoder setBuffer:top_b offset:0 atIndex:7];
                    if (use_coeff_buffers) {
                        [encoder setBuffer:ax_b offset:0 atIndex:8];
                        [encoder setBuffer:by_b offset:0 atIndex:9];
                        [encoder setBuffer:cx_b offset:0 atIndex:10];
                        [encoder setBuffer:dyc_b offset:0 atIndex:11];
                        [encoder setBuffer:center_b offset:0 atIndex:12];
                        [encoder setBuffer:rhs_b offset:0 atIndex:13];
                    } else {
                        [encoder setBytes:&ax_const length:sizeof(float) atIndex:8];
                        [encoder setBytes:&by_const length:sizeof(float) atIndex:9];
                        [encoder setBytes:&cx_const length:sizeof(float) atIndex:10];
                        [encoder setBytes:&dyc_const length:sizeof(float) atIndex:11];
                        [encoder setBytes:&center_const length:sizeof(float) atIndex:12];
                        [encoder setBytes:&f_const length:sizeof(float) atIndex:13];
                    }
                    Dispatch2D(encoder, nx, ny, tg);
                    std::swap(grid_buf, next_buf);
                } else { // SOR / Gauss-Seidel
                    const float omega = input.solver.method == SolveMethod::SOR ? (float)input.solver.sor_omega : 1.0f;
                    [encoder setComputePipelineState:boundary_pipeline];
                    [encoder setBuffer:grid_buf offset:0 atIndex:0];
                    [encoder setBuffer:domain_b offset:0 atIndex:1];
                    [encoder setBuffer:left_b offset:0 atIndex:2];
                    [encoder setBuffer:right_b offset:0 atIndex:3];
                    [encoder setBuffer:bottom_b offset:0 atIndex:4];
                    [encoder setBuffer:top_b offset:0 atIndex:5];
                    Dispatch2D(encoder, nx, ny, tg);
                    
                    uint parity = 0;
                    [encoder setComputePipelineState:rbgs_pipeline];
                    [encoder setBuffer:grid_buf offset:0 atIndex:0];
                    [encoder setBuffer:delta_buf offset:0 atIndex:1];
                    [encoder setBuffer:domain_b offset:0 atIndex:2];
                    if (use_coeff_buffers) {
                        [encoder setBuffer:ax_b offset:0 atIndex:3];
                        [encoder setBuffer:by_b offset:0 atIndex:4];
                        [encoder setBuffer:cx_b offset:0 atIndex:5];
                        [encoder setBuffer:dyc_b offset:0 atIndex:6];
                        [encoder setBuffer:center_b offset:0 atIndex:7];
                        [encoder setBuffer:rhs_b offset:0 atIndex:8];
                    } else {
                        [encoder setBytes:&ax_const length:sizeof(float) atIndex:3];
                        [encoder setBytes:&by_const length:sizeof(float) atIndex:4];
                        [encoder setBytes:&cx_const length:sizeof(float) atIndex:5];
                        [encoder setBytes:&dyc_const length:sizeof(float) atIndex:6];
                        [encoder setBytes:&center_const length:sizeof(float) atIndex:7];
                        [encoder setBytes:&f_const length:sizeof(float) atIndex:8];
                    }
                    [encoder setBytes:&omega length:sizeof(float) atIndex:9];
                    [encoder setBytes:&parity length:sizeof(uint) atIndex:10];
                    Dispatch2D(encoder, nx, ny, tg);
                    
                    parity = 1;
                    [encoder setBytes:&parity length:sizeof(uint) atIndex:10];
                    Dispatch2D(encoder, nx, ny, tg);
                }
            }
            iter += batch;

            // Reduction
            uint32_t reduce_count = total;
            reduce_in = delta_buf;  // Reset for reduction
            while(reduce_count > 1) {
                 uint32_t groups = (reduce_count + kReduceGroupSize - 1) / kReduceGroupSize;
                 [encoder setComputePipelineState:reduce_pipeline];
                 [encoder setBuffer:reduce_in offset:0 atIndex:0];
                 [encoder setBuffer:reduce_a offset:0 atIndex:1];
                 [encoder setBytes:&reduce_count length:sizeof(uint32_t) atIndex:2];
                 Dispatch1D(encoder, reduce_count, kReduceGroupSize);
                 reduce_count = groups;
                 reduce_in = reduce_a;
            }
            
            } @finally {
                [encoder endEncoding];
            }
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:reduce_in sourceOffset:0 toBuffer:reduce_res destinationOffset:0 size:sizeof(float)];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        float max_delta = *(static_cast<const float*>([reduce_res contents]));
        if (max_delta < input.solver.tol) break;
    }

    std::vector<double> grid_host(total);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:grid_buf sourceOffset:0 toBuffer:host_res_buf destinationOffset:0 size:sizeof(float) * total];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    const float* grid_ptr = static_cast<const float*>([host_res_buf contents]);
    for(int i = 0; i < total; ++i) grid_host[i] = grid_ptr[i];

    return {"", grid_host};
}
