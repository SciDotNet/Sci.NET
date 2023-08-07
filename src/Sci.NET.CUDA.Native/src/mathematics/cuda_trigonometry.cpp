//
// Created by reece on 01/08/2023.
//

#include "cuda_trigonometry.h"
#include "kernels/trigonometry_kernels.cuh"

sdnApiStatusCode sin_fp32(float *a, float *b, int64_t n) {
    auto result = sin_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sin_fp64(double *a, double *b, int64_t n) {
    auto result = sin_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cos_fp32(float *a, float *b, int64_t n) {
    auto result = cos_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cos_fp64(double *a, double *b, int64_t n) {
    auto result = cos_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tan_fp32(float *a, float *b, int64_t n) {
    auto result = tan_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tan_fp64(double *a, double *b, int64_t n) {
    auto result = tan_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sin2_fp32(float *a, float *b, int64_t n) {
    auto result = sin2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sin2_fp64(double *a, double *b, int64_t n) {
    auto result = sin2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cos2_fp32(float *a, float *b, int64_t n) {
    auto result = cos2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cos2_fp64(double *a, double *b, int64_t n) {
    auto result = cos2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tan2_fp32(float *a, float *b, int64_t n) {
    auto result = tan2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tan2_fp64(double *a, double *b, int64_t n) {
    auto result = tan2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sinh_fp32(float *a, float *b, int64_t n) {
    auto result = sinh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sinh_fp64(double *a, double *b, int64_t n) {
    auto result = sinh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cosh_fp32(float *a, float *b, int64_t n) {
    auto result = cosh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cosh_fp64(double *a, double *b, int64_t n) {
    auto result = cosh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tanh_fp32(float *a, float *b, int64_t n) {
    auto result = tanh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tanh_fp64(double *a, double *b, int64_t n) {
    auto result = tanh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sinh2_fp32(float *a, float *b, int64_t n) {
    auto result = sinh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sinh2_fp64(double *a, double *b, int64_t n) {
    auto result = sinh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cosh2_fp32(float *a, float *b, int64_t n) {
    auto result = cosh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cosh2_fp64(double *a, double *b, int64_t n) {
    auto result = cosh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tanh2_fp32(float *a, float *b, int64_t n) {
    auto result = tanh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode tanh2_fp64(double *a, double *b, int64_t n) {
    auto result = tanh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asin_fp32(float *a, float *b, int64_t n) {
    auto result = asin_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asin_fp64(double *a, double *b, int64_t n) {
    auto result = asin_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acos_fp32(float *a, float *b, int64_t n) {
    auto result = acos_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acos_fp64(double *a, double *b, int64_t n) {
    auto result = acos_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atan_fp32(float *a, float *b, int64_t n) {
    auto result = atan_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atan_fp64(double *a, double *b, int64_t n) {
    auto result = atan_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asin2_fp32(float *a, float *b, int64_t n) {
    auto result = asin2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asin2_fp64(double *a, double *b, int64_t n) {
    auto result = asin2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acos2_fp32(float *a, float *b, int64_t n) {
    auto result = acos2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acos2_fp64(double *a, double *b, int64_t n) {
    auto result = acos2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atan2_fp32(float *a, float *b, int64_t n) {
    auto result = atan2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atan2_fp64(double *a, double *b, int64_t n) {
    auto result = atan2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asinh_fp32(float *a, float *b, int64_t n) {
    auto result = asinh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asinh_fp64(double *a, double *b, int64_t n) {
    auto result = asinh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acosh_fp32(float *a, float *b, int64_t n) {
    auto result = acosh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acosh_fp64(double *a, double *b, int64_t n) {
    auto result = acosh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atanh_fp32(float *a, float *b, int64_t n) {
    auto result = atanh_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atanh_fp64(double *a, double *b, int64_t n) {
    auto result = atanh_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asinh2_fp32(float *a, float *b, int64_t n) {
    auto result = asinh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asinh2_fp64(double *a, double *b, int64_t n) {
    auto result = asinh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acosh2_fp32(float *a, float *b, int64_t n) {
    auto result = acosh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acosh2_fp64(double *a, double *b, int64_t n) {
    auto result = acosh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atanh2_fp32(float *a, float *b, int64_t n) {
    auto result = atanh2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode atanh2_fp64(double *a, double *b, int64_t n) {
    auto result = atanh2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csc_fp32(float *a, float *b, int64_t n) {
    auto result = csc_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csc_fp64(double *a, double *b, int64_t n) {
    auto result = csc_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sec_fp32(float *a, float *b, int64_t n) {
    auto result = sec_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sec_fp64(double *a, double *b, int64_t n) {
    auto result = sec_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cot_fp32(float *a, float *b, int64_t n) {
    auto result = cot_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cot_fp64(double *a, double *b, int64_t n) {
    auto result = cot_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csc2_fp32(float *a, float *b, int64_t n) {
    auto result = csc2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csc2_fp64(double *a, double *b, int64_t n) {
    auto result = csc2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sec2_fp32(float *a, float *b, int64_t n) {
    auto result = sec2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sec2_fp64(double *a, double *b, int64_t n) {
    auto result = sec2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cot2_fp32(float *a, float *b, int64_t n) {
    auto result = cot2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode cot2_fp64(double *a, double *b, int64_t n) {
    auto result = cot2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csch_fp32(float *a, float *b, int64_t n) {
    auto result = csch_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csch_fp64(double *a, double *b, int64_t n) {
    auto result = csch_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sech_fp32(float *a, float *b, int64_t n) {
    auto result = sech_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sech_fp64(double *a, double *b, int64_t n) {
    auto result = sech_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode coth_fp32(float *a, float *b, int64_t n) {
    auto result = coth_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode coth_fp64(double *a, double *b, int64_t n) {
    auto result = coth_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csch2_fp32(float *a, float *b, int64_t n) {
    auto result = csch2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode csch2_fp64(double *a, double *b, int64_t n) {
    auto result = csch2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sech2_fp32(float *a, float *b, int64_t n) {
    auto result = sech2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode sech2_fp64(double *a, double *b, int64_t n) {
    auto result = sech2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode coth2_fp32(float *a, float *b, int64_t n) {
    auto result = coth2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode coth2_fp64(double *a, double *b, int64_t n) {
    auto result = coth2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsc_fp32(float *a, float *b, int64_t n) {
    auto result = acsc_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsc_fp64(double *a, double *b, int64_t n) {
    auto result = acsc_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asec_fp32(float *a, float *b, int64_t n) {
    auto result = asec_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asec_fp64(double *a, double *b, int64_t n) {
    auto result = asec_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acot_fp32(float *a, float *b, int64_t n) {
    auto result = acot_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acot_fp64(double *a, double *b, int64_t n) {
    auto result = acot_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsc2_fp32(float *a, float *b, int64_t n) {
    auto result = acsc2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsc2_fp64(double *a, double *b, int64_t n) {
    auto result = acsc2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asec2_fp32(float *a, float *b, int64_t n) {
    auto result = asec2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asec2_fp64(double *a, double *b, int64_t n) {
    auto result = asec2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acot2_fp32(float *a, float *b, int64_t n) {
    auto result = acot2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acot2_fp64(double *a, double *b, int64_t n) {
    auto result = acot2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsch_fp32(float *a, float *b, int64_t n) {
    auto result = acsch_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsch_fp64(double *a, double *b, int64_t n) {
    auto result = acsch_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asech_fp32(float *a, float *b, int64_t n) {
    auto result = asech_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asech_fp64(double *a, double *b, int64_t n) {
    auto result = asech_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acoth_fp32(float *a, float *b, int64_t n) {
    auto result = acoth_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acoth_fp64(double *a, double *b, int64_t n) {
    auto result = acoth_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsch2_fp32(float *a, float *b, int64_t n) {
    auto result = acsch2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acsch2_fp64(double *a, double *b, int64_t n) {
    auto result = acsch2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asech2_fp32(float *a, float *b, int64_t n) {
    auto result = asech2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode asech2_fp64(double *a, double *b, int64_t n) {
    auto result = asech2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acoth2_fp32(float *a, float *b, int64_t n) {
    auto result = acoth2_fp32_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode acoth2_fp64(double *a, double *b, int64_t n) {
    auto result = acoth2_fp64_invoke(a, b, n);
    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}
