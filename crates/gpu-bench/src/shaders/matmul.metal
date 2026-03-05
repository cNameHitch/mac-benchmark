#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

kernel void matmul(device const float *A [[buffer(0)]],
                   device const float *B [[buffer(1)]],
                   device float *C [[buffer(2)]],
                   constant uint &N [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]],
                   uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;

    for (uint t = 0; t < N; t += TILE) {
        // Load tile of A
        uint a_col = t + lid.x;
        if (row < N && a_col < N)
            As[lid.y][lid.x] = A[row * N + a_col];
        else
            As[lid.y][lid.x] = 0.0f;

        // Load tile of B
        uint b_row = t + lid.y;
        if (b_row < N && col < N)
            Bs[lid.y][lid.x] = B[b_row * N + col];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            sum = fma(As[lid.y][k], Bs[k][lid.x], sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
