#include <metal_stdlib>
using namespace metal;

kernel void fp32_throughput(device float *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    float a0 = 1.0f + float(tid) * 1e-7f;
    float a1 = a0 + 1e-7f;
    float a2 = a0 + 2e-7f;
    float a3 = a0 + 3e-7f;

    for (uint i = 0; i < 1024; i++) {
        a0 = fma(a0, 1.0000001f, 0.0000001f);
        a1 = fma(a1, 1.0000001f, 0.0000001f);
        a2 = fma(a2, 1.0000001f, 0.0000001f);
        a3 = fma(a3, 1.0000001f, 0.0000001f);
    }

    out[tid] = a0 + a1 + a2 + a3;
}

kernel void fp16_throughput(device half *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    half a0 = half(1.0h + half(tid & 0xFF) * 1e-4h);
    half a1 = a0 + 1e-4h;
    half a2 = a0 + 2e-4h;
    half a3 = a0 + 3e-4h;

    for (uint i = 0; i < 1024; i++) {
        a0 = fma(a0, half(1.001h), half(0.001h));
        a1 = fma(a1, half(1.001h), half(0.001h));
        a2 = fma(a2, half(1.001h), half(0.001h));
        a3 = fma(a3, half(1.001h), half(0.001h));
    }

    out[tid] = a0 + a1 + a2 + a3;
}

kernel void int32_throughput(device uint *out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
    uint a0 = tid + 1u;
    uint a1 = tid + 2u;
    uint a2 = tid + 3u;
    uint a3 = tid + 4u;

    for (uint i = 0; i < 1024; i++) {
        a0 = a0 * 3u + 7u;
        a1 = a1 * 3u + 7u;
        a2 = a2 * 3u + 7u;
        a3 = a3 * 3u + 7u;
    }

    out[tid] = a0 + a1 + a2 + a3;
}
