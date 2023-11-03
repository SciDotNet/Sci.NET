//
// Created by reece on 01/08/2023.
//

#include "trigonometry_kernels.cuh"

__global__ void sin_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = sinf(a[index]);
    }
}

__global__ void sin_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = sin(a[index]);
    }
}

__global__ void cos_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = cosf(a[index]);
    }
}

__global__ void cos_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = cos(a[index]);
    }
}

__global__ void tan_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = tanf(a[index]);
    }
}

__global__ void tan_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = tan(a[index]);
    }
}

__global__ void sin2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float sin_a = sinf(a[index]);

        b[index] = sin_a * sin_a;
    }
}

__global__ void sin2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double sin_a = sin(a[index]);

        b[index] = sin_a * sin_a;
    }
}

__global__ void cos2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float cos_a = cosf(a[index]);

        b[index] = cos_a * cos_a;
    }
}

__global__ void cos2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double cos_a = cos(a[index]);

        b[index] = cos_a * cos_a;
    }
}

__global__ void tan2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float tan_a = tanf(a[index]);

        b[index] = tan_a * tan_a;
    }
}

__global__ void tan2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double tan_a = tan(a[index]);

        b[index] = tan_a * tan_a;
    }
}

__global__ void sinh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = sinhf(a[index]);
    }
}

__global__ void sinh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = sinh(a[index]);
    }
}

__global__ void cosh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = coshf(a[index]);
    }
}

__global__ void cosh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = cosh(a[index]);
    }
}

__global__ void tanh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = tanhf(a[index]);
    }
}

__global__ void tanh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = tanh(a[index]);
    }
}

__global__ void sinh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float sinh_a = sinhf(a[index]);

        b[index] = sinh_a * sinh_a;
    }
}

__global__ void sinh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double sinh_a = sinh(a[index]);

        b[index] = sinh_a * sinh_a;
    }
}

__global__ void cosh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float cosh_a = coshf(a[index]);

        b[index] = cosh_a * cosh_a;
    }
}

__global__ void cosh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double cosh_a = cosh(a[index]);

        b[index] = cosh_a * cosh_a;
    }
}

__global__ void tanh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        float tanh_a = tanhf(a[index]);

        b[index] = tanh_a * tanh_a;
    }
}

__global__ void tanh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {

        double tanh_a = tanh(a[index]);

        b[index] = tanh_a * tanh_a;
    }
}

__global__ void asin_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = asinf(a[index]);
    }
}

__global__ void asin_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = asin(a[index]);
    }
}

__global__ void acos_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = acosf(a[index]);
    }
}

__global__ void acos_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = acos(a[index]);
    }
}

__global__ void atan_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = atanf(a[index]);
    }
}

__global__ void atan_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = atan(a[index]);
    }
}

__global__ void asin2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {

        float asin_a = asinf(a[index]);

        b[index] = asin_a * asin_a;
    }
}

__global__ void asin2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {

        double asin_a = asin(a[index]);

        b[index] = asin_a * asin_a;
    }
}

__global__ void acos2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {

        float acos_a = acosf(a[index]);

        b[index] = acos_a * acos_a;
    }
}

__global__ void acos2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {

        double acos_a = acos(a[index]);

        b[index] = acos_a * acos_a;
    }
}

__global__ void atan2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = 1.0f / atanf(a[index]);
    }
}

__global__ void atan2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        b[index] = 1.0 / atan(a[index]);
    }
}

__global__ void asinh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = asinhf(a[index]);
    }
}

__global__ void asinh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = asinh(a[index]);
    }
}

__global__ void acosh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= 1.0) {
        b[index] = acoshf(a[index]);
    }
}

__global__ void acosh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= 1.0) {
        b[index] = acosh(a[index]);
    }
}

__global__ void atanh_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = atanhf(a[index]);
    }
}

__global__ void atanh_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= -1.0 && a[index] <= 1.0) {
        b[index] = atanh(a[index]);
    }
}

__global__ void asinh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && fabsf(a[index]) <= 1.0) {

        float asinh_a = asinhf(a[index]);

        b[index] = asinh_a * asinh_a;
    }
}

__global__ void asinh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && fabs(a[index]) <= 1.0) {

        double asinh_a = asinh(a[index]);

        b[index] = asinh_a * asinh_a;
    }
}

__global__ void acosh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= 1.0) {

        float acosh_a = acoshf(a[index]);

        b[index] = acosh_a * acosh_a;
    }
}

__global__ void acosh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && a[index] >= 1.0) {

        double acosh_a = acosh(a[index]);

        b[index] = acosh_a * acosh_a;
    }
}

__global__ void atanh2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && fabsf(a[index]) <= 1.0) {

        float atanh_a = atanhf(a[index]);

        b[index] = atanh_a * atanh_a;
    }
}

__global__ void atanh2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && fabs(a[index]) <= 1.0) {

        double atanh_a = atanh(a[index]);

        b[index] = atanh_a * atanh_a;
    }
}

__global__ void csc_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && sinf(a[index]) != 0.0) {
        b[index] = 1.0f / sinf(a[index]);
    }
}

__global__ void csc_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && sin(a[index]) != 0.0) {
        b[index] = 1.0 / sin(a[index]);
    }
}

__global__ void sec_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && cosf(a[index]) != 0.0) {
        b[index] = 1.0f / cosf(a[index]);
    }
}

__global__ void sec_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n && cos(a[index]) != 0.0) {
        b[index] = 1.0 / cos(a[index]);
    }
}

__global__ void cot_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float tan_a = tanf(a[index]);

    if (index < n && tan_a != 0.0) {
        b[index] = 1.0f / tan_a;
    }
}

__global__ void cot_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double tan_a = tan(a[index]);

    if (index < n && tan_a != 0.0) {
        b[index] = 1.0 / tan_a;
    }
}

__global__ void csc2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sin_a = sinf(a[index]);

    if (index < n && sin_a != 0.0) {
        b[index] = 1.0f / (sin_a * sin_a);
    }
}

__global__ void csc2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sin_a = sin(a[index]);

    if (index < n && sin_a != 0.0) {
        b[index] = 1.0 / (sin_a * sin_a);
    }
}

__global__ void sec2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float cos_a = cosf(a[index]);

    if (index < n && cos_a != 0.0) {
        b[index] = 1.0f / (cos_a * cos_a);
    }
}

__global__ void sec2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double cos_a = cos(a[index]);

    if (index < n && cos_a != 0.0) {
        b[index] = 1.0 / (cos_a * cos_a);
    }
}

__global__ void cot2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float tan_a = tanf(a[index]);

    if (index < n && tan_a != 0.0) {
        b[index] = 1.0f / (tan_a * tan_a);
    }
}

__global__ void cot2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double tan_a = tan(a[index]);

    if (index < n && tan_a != 0.0) {
        b[index] = 1.0 / (tan_a * tan_a);
    }
}

__global__ void csch_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sinh_a = sinhf(a[index]);

    if (index < n && sinh_a != 0.0) {
        b[index] = 1.0f / sinh_a;
    }
}

__global__ void csch_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sinh_a = sinh(a[index]);

    if (index < n && sinh_a != 0.0) {
        b[index] = 1.0 / sinh_a;
    }
}

__global__ void sech_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float cosh_a = coshf(a[index]);

    if (index < n && cosh_a != 0.0) {
        b[index] = 1.0f / cosh_a;
    }
}

__global__ void sech_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double cosh_a = cosh(a[index]);

    if (index < n && cosh_a != 0.0) {
        b[index] = 1.0 / cosh_a;
    }
}

__global__ void coth_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float tanh_a = tanhf(a[index]);

    if (index < n && tanh_a != 0.0) {
        b[index] = 1.0f / tanh_a;
    }
}

__global__ void coth_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double tanh_a = tanh(a[index]);

    if (index < n && tanh_a != 0.0) {
        b[index] = 1.0 / tanh_a;
    }
}

__global__ void csc2h_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sinh_a = sinhf(a[index]);

    if (index < n && sinh_a != 0.0) {
        b[index] = 1.0f / (sinh_a * sinh_a);
    }
}

__global__ void csc2h_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sinh_a = sinh(a[index]);

    if (index < n && sinh_a != 0.0) {
        b[index] = 1.0 / (sinh_a * sinh_a);
    }
}

__global__ void sech2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float cosh_a = coshf(a[index]);

    if (index < n && cosh_a != 0.0) {
        b[index] = 1.0f / (cosh_a * cosh_a);
    }
}

__global__ void sech2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double cosh_a = cosh(a[index]);

    if (index < n && cosh_a != 0.0) {
        b[index] = 1.0 / (cosh_a * cosh_a);
    }
}

__global__ void coth2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float tanh_a = tanhf(a[index]);

    if (index < n && tanh_a != 0.0) {
        b[index] = 1.0f / (tanh_a * tanh_a);
    }
}

__global__ void coth2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double tanh_a = tanh(a[index]);

    if (index < n && tanh_a != 0.0) {
        b[index] = 1.0 / (tanh_a * tanh_a);
    }
}

__global__ void acsc_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float csc_a = 1.0f / a[index];

    if (index < n && csc_a >= -1.0 && csc_a <= 1.0) {
        b[index] = asinf(csc_a);
    }
}

__global__ void acsc_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double csc_a = 1.0 / a[index];

    if (index < n && csc_a >= -1.0 && csc_a <= 1.0) {
        b[index] = asin(csc_a);
    }
}

__global__ void asec_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sec_a = 1.0f / a[index];

    if (index < n && sec_a >= -1.0 && sec_a <= 1.0) {
        b[index] = acosf(sec_a);
    }
}

__global__ void asec_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sec_a = 1.0 / a[index];

    if (index < n && sec_a >= -1.0 && sec_a <= 1.0) {
        b[index] = acos(sec_a);
    }
}

__global__ void acot_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float cot_a = 1.0f / a[index];

    if (index < n) {
        b[index] = atanf(cot_a);
    }
}

__global__ void acot_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double cot_a = 1.0 / a[index];

    if (index < n) {
        b[index] = atan(cot_a);
    }
}

__global__ void acsc2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float csc_a = 1.0f / a[index];

    if (index < n && csc_a >= -1.0 && csc_a <= 1.0) {

        float asin_csc_a = asinf(csc_a);

        b[index] = asin_csc_a * asin_csc_a;
    }
}

__global__ void acsc2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double csc_a = 1.0 / a[index];

    if (index < n && csc_a >= -1.0 && csc_a <= 1.0) {

        double asin_csc_a = asin(csc_a);

        b[index] = asin_csc_a * asin_csc_a;
    }
}

__global__ void asec2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sec_a = 1.0f / a[index];

    if (index < n && sec_a >= -1.0 && sec_a <= 1.0) {

        float acos_sec_a = acosf(sec_a);

        b[index] = acos_sec_a * acos_sec_a;
    }
}

__global__ void asec2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sec_a = 1.0 / a[index];

    if (index < n && sec_a >= -1.0 && sec_a <= 1.0) {

        double acos_sec_a = acos(sec_a);

        b[index] = acos_sec_a * acos_sec_a;
    }
}

__global__ void acot2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float cot_a = 1.0f / a[index];

    if (index < n) {

        float atan_cot_a = atanf(cot_a);

        b[index] = atan_cot_a * atan_cot_a;
    }
}

__global__ void acot2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double cot_a = 1.0 / a[index];

    if (index < n) {

        double atan_cot_a = atan(cot_a);

        b[index] = atan_cot_a * atan_cot_a;
    }
}

__global__ void acsch_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float csch_a = 1.0f / sinh(a[index]);

    if (index < n && csch_a != 0.0) {
        b[index] = logf(csch_a + sqrtf(csch_a * csch_a + 1.0f));
    }
}

__global__ void acsch_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double csch_a = 1.0 / sinh(a[index]);

    if (index < n && csch_a != 0.0) {
        b[index] = log(csch_a + sqrt(csch_a * csch_a + 1.0));
    }
}

__global__ void asech_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sech_a = 1.0f / cosh(a[index]);

    if (index < n && sech_a != 0.0) {
        b[index] = logf(sech_a + sqrtf(sech_a * sech_a - 1.0f));
    }
}

__global__ void asech_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sech_a = 1.0 / cosh(a[index]);

    if (index < n && sech_a != 0.0) {
        b[index] = log(sech_a + sqrt(sech_a * sech_a - 1.0));
    }
}

__global__ void acoth_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float coth_a = 1.0f / tanh(a[index]);

    if (index < n && coth_a != 0.0) {
        b[index] = 0.5f * logf((coth_a + 1.0f) / (coth_a - 1.0f));
    }
}

__global__ void acoth_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double coth_a = 1.0 / tanh(a[index]);

    if (index < n && coth_a != 0.0) {
        b[index] = 0.5 * log((coth_a + 1.0) / (coth_a - 1.0));
    }
}

__global__ void acsch2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float csch_a = 1.0f / sinh(a[index]);

    if (index < n && csch_a != 0.0) {

        float log_csch_a = logf(csch_a + sqrtf(csch_a * csch_a + 1.0f));

        b[index] = log_csch_a * log_csch_a;
    }
}

__global__ void acsch2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double csch_a = 1.0 / sinh(a[index]);

    if (index < n && csch_a != 0.0) {

        double log_csch_a = log(csch_a + sqrt(csch_a * csch_a + 1.0));

        b[index] = log_csch_a * log_csch_a;
    }
}

__global__ void asech2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sech_a = 1.0f / cosh(a[index]);

    if (index < n && sech_a != 0.0) {

        float log_sech_a = logf(sech_a + sqrtf(sech_a * sech_a - 1.0f));

        b[index] = log_sech_a * log_sech_a;
    }
}

__global__ void asech2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double sech_a = 1.0 / cosh(a[index]);

    if (index < n && sech_a != 0.0) {

        double log_sech_a = log(sech_a + sqrt(sech_a * sech_a - 1.0));

        b[index] = log_sech_a * log_sech_a;
    }
}

__global__ void acoth2_fp32_kernel(float *a, float *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float coth_a = 1.0f / tanh(a[index]);

    if (index < n && coth_a != 0.0) {

        float log_coth_a = 0.5f * logf((coth_a + 1.0f) / (coth_a - 1.0f));

        b[index] = log_coth_a * log_coth_a;
    }
}

__global__ void acoth2_fp64_kernel(double *a, double *b, int64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double coth_a = 1.0 / tanh(a[index]);

    if (index < n && coth_a != 0.0) {

        double log_coth_a = 0.5 * log((coth_a + 1.0) / (coth_a - 1.0));

        b[index] = log_coth_a * log_coth_a;
    }
}



bool sin_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sin_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sin_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sin_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}


bool cos_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cos_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cos_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cos_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tan_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tan_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}


bool tan_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tan_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sin2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sin2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sin2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sin2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cos2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cos2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cos2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cos2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tan2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tan2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tan2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tan2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sinh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sinh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sinh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sinh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cosh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cosh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cosh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cosh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tanh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tanh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sinh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sinh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sinh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sinh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cosh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cosh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cosh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cosh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tanh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tanh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool tanh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tanh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asin_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    asin_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asin_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    asin_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acos_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acos_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acos_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acos_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atan_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atan_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atan_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atan_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asin2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asin2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asin2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asin2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acos2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acos2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acos2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acos2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atan2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atan2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atan2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atan2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asinh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asinh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asinh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asinh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acosh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acosh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acosh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acosh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atanh_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    atanh_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atanh_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    atanh_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asinh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    asinh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asinh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    asinh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acosh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acosh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acosh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acosh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atanh2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atanh2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool atanh2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    atanh2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csc_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csc_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csc_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csc_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sec_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sec_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sec_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sec_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cot_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cot_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cot_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cot_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csc2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csc2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csc2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csc2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sec2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sec2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sec2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sec2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cot2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cot2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool cot2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cot2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csch_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csch_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csch_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    csch_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sech_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sech_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sech_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sech_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}


bool coth_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    coth_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool coth_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    coth_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csch2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acsch2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool csch2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acsch2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sech2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sech2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool sech2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sech2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool coth2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    coth2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool coth2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    coth2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsc_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acsc_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsc_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acsc_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asec_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    asec_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asec_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asec_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acot_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acot_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acot_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acot_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsc2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acsc2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsc2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acsc2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asec2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asec2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asec2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asec2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acot2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acot2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acot2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acot2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsch_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acsch_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsch_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acsch_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asech_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asech_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asech_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    asech_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acoth_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    acoth_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acoth_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    acoth_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsch2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    acsch2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acsch2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    acsch2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asech2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    asech2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool asech2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    asech2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acoth2_fp32_invoke(float *a, float *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    acoth2_fp32_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool acoth2_fp64_invoke(double *a, double *b, int64_t n) {

    int threads = 256;
    int blocks = (n + threads - 1);
    acoth2_fp64_kernel<<<blocks, threads>>>(a, b, n);

    return cudaDeviceSynchronize() == cudaSuccess;
}
