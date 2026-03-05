#include <metal_stdlib>
using namespace metal;

kernel void buffer_read(device const float4 *src [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint tid [[thread_position_in_grid]],
                        uint total [[threads_per_grid]]) {
    float4 sum = float4(0.0f);
    for (uint i = tid; i < count; i += total) {
        sum += src[i];
    }
    out[tid] = sum.x + sum.y + sum.z + sum.w;
}

kernel void buffer_write(device float4 *dst [[buffer(0)]],
                         constant uint &count [[buffer(1)]],
                         uint tid [[thread_position_in_grid]],
                         uint total [[threads_per_grid]]) {
    float4 val = float4(float(tid));
    for (uint i = tid; i < count; i += total) {
        dst[i] = val;
    }
}
