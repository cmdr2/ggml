#include "conv.cuh"

constexpr int BLOCK_DIM_Y = 16;
constexpr int BLOCK_DIM_X = 16;

__global__ void conv2d_kernel(
    const float* __restrict__ input,   // [B, IC, H, W]
    const float* __restrict__ kernel,  // [KW, KH, IC, OC]
    float* __restrict__ output,        // [B, OC, OH, OW]
    int B, int IC, int H, int W,
    int OC, int OH, int OW,
    int kH, int kW,
    int strideY, int strideX,
    int padY, int padX,
    int dilY, int dilX)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z / OC;
    const int oc = blockIdx.z % OC;

    if (out_x >= OW || out_y >= OH || b >= B) return;

    const int in_x = out_x * strideX - padX;
    const int in_y = out_y * strideY - padY;

    float sum = 0.0f;

    for (int ic = 0; ic < IC; ++ic) {
        for (int ky = 0; ky < kH; ++ky) {
            for (int kx = 0; kx < kW; ++kx) {
                int iy = in_y + ky * dilY;
                int ix = in_x + kx * dilX;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    float input_val = input[b * IC * H * W + ic * H * W + iy * W + ix];
                    float weight_val = kernel[kx + ky * kW + ic * kW * kH + oc * kW * kH * IC];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    output[b * OC * OH * OW + oc * OH * OW + out_y * OW + out_x] = sum;
}

void ggml_cuda_op_conv_2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const int64_t IC = src1->ne[2];
    const int64_t IH = src1->ne[1];
    const int64_t IW = src1->ne[0];

    const int64_t OC = src0->ne[3];
    const int64_t KH = src0->ne[1];
    const int64_t KW = src0->ne[0];

    const int64_t B  = dst->ne[3];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, B * OC);
    conv2d_kernel<<<grid, block>>>(
                (const float *)src1->data, (const float *)src0->data, (float *)dst->data,
                B, IC, IH, IW,
                OC, OH, OW,
                KH, KW, s1, s0,
                p1, p0, d1, d0);
}
