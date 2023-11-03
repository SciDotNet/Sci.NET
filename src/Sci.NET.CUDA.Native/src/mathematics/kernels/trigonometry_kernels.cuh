//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_TRIGONOMETRY_KERNELS_H
#define SCI_NET_NATIVE_TRIGONOMETRY_KERNELS_H

bool sin_fp32_invoke(float *tensor, float *result, int64_t n);
bool sin_fp64_invoke(double *tensor, double *result, int64_t n);
bool cos_fp32_invoke(float *tensor, float *result, int64_t n);
bool cos_fp64_invoke(double *tensor, double *result, int64_t n);
bool tan_fp32_invoke(float *tensor, float *result, int64_t n);
bool tan_fp64_invoke(double *tensor, double *result, int64_t n);
bool sin2_fp32_invoke(float *tensor, float *result, int64_t n);
bool sin2_fp64_invoke(double *tensor, double *result, int64_t n);
bool cos2_fp32_invoke(float *tensor, float *result, int64_t n);
bool cos2_fp64_invoke(double *tensor, double *result, int64_t n);
bool tan2_fp32_invoke(float *tensor, float *result, int64_t n);
bool tan2_fp64_invoke(double *tensor, double *result, int64_t n);
bool sinh_fp32_invoke(float *tensor, float *result, int64_t n);
bool sinh_fp64_invoke(double *tensor, double *result, int64_t n);
bool cosh_fp32_invoke(float *tensor, float *result, int64_t n);
bool cosh_fp64_invoke(double *tensor, double *result, int64_t n);
bool tanh_fp32_invoke(float *tensor, float *result, int64_t n);
bool tanh_fp64_invoke(double *tensor, double *result, int64_t n);
bool sinh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool sinh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool cosh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool cosh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool tanh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool tanh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool asin_fp32_invoke(float *tensor, float *result, int64_t n);
bool asin_fp64_invoke(double *tensor, double *result, int64_t n);
bool acos_fp32_invoke(float *tensor, float *result, int64_t n);
bool acos_fp64_invoke(double *tensor, double *result, int64_t n);
bool atan_fp32_invoke(float *tensor, float *result, int64_t n);
bool atan_fp64_invoke(double *tensor, double *result, int64_t n);
bool asin2_fp32_invoke(float *tensor, float *result, int64_t n);
bool asin2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acos2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acos2_fp64_invoke(double *tensor, double *result, int64_t n);
bool atan2_fp32_invoke(float *tensor, float *result, int64_t n);
bool atan2_fp64_invoke(double *tensor, double *result, int64_t n);
bool asinh_fp32_invoke(float *tensor, float *result, int64_t n);
bool asinh_fp64_invoke(double *tensor, double *result, int64_t n);
bool acosh_fp32_invoke(float *tensor, float *result, int64_t n);
bool acosh_fp64_invoke(double *tensor, double *result, int64_t n);
bool atanh_fp32_invoke(float *tensor, float *result, int64_t n);
bool atanh_fp64_invoke(double *tensor, double *result, int64_t n);
bool asinh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool asinh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acosh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acosh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool atanh2_fp32_invoke(float *tensor, float *result, int64_t n);
bool atanh2_fp64_invoke(double *tensor, double *result, int64_t n);
bool csc_fp32_invoke(float *tensor, float *result, int64_t n);
bool csc_fp64_invoke(double *tensor, double *result, int64_t n);
bool sec_fp32_invoke(float *tensor, float *result, int64_t n);
bool sec_fp64_invoke(double *tensor, double *result, int64_t n);
bool cot_fp32_invoke(float *tensor, float *result, int64_t n);
bool cot_fp64_invoke(double *tensor, double *result, int64_t n);
bool csc2_fp32_invoke(float *tensor, float *result, int64_t n);
bool csc2_fp64_invoke(double *tensor, double *result, int64_t n);
bool sec2_fp32_invoke(float *tensor, float *result, int64_t n);
bool sec2_fp64_invoke(double *tensor, double *result, int64_t n);
bool cot2_fp32_invoke(float *tensor, float *result, int64_t n);
bool cot2_fp64_invoke(double *tensor, double *result, int64_t n);
bool csch_fp32_invoke(float *tensor, float *result, int64_t n);
bool csch_fp64_invoke(double *tensor, double *result, int64_t n);
bool sech_fp32_invoke(float *tensor, float *result, int64_t n);
bool sech_fp64_invoke(double *tensor, double *result, int64_t n);
bool coth_fp32_invoke(float *tensor, float *result, int64_t n);
bool coth_fp64_invoke(double *tensor, double *result, int64_t n);
bool csch2_fp32_invoke(float *tensor, float *result, int64_t n);
bool csch2_fp64_invoke(double *tensor, double *result, int64_t n);
bool sech2_fp32_invoke(float *tensor, float *result, int64_t n);
bool sech2_fp64_invoke(double *tensor, double *result, int64_t n);
bool coth2_fp32_invoke(float *tensor, float *result, int64_t n);
bool coth2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acsc_fp32_invoke(float *tensor, float *result, int64_t n);
bool acsc_fp64_invoke(double *tensor, double *result, int64_t n);
bool asec_fp32_invoke(float *tensor, float *result, int64_t n);
bool asec_fp64_invoke(double *tensor, double *result, int64_t n);
bool acot_fp32_invoke(float *tensor, float *result, int64_t n);
bool acot_fp64_invoke(double *tensor, double *result, int64_t n);
bool acsc2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acsc2_fp64_invoke(double *tensor, double *result, int64_t n);
bool asec2_fp32_invoke(float *tensor, float *result, int64_t n);
bool asec2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acot2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acot2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acsch_fp32_invoke(float *tensor, float *result, int64_t n);
bool acsch_fp64_invoke(double *tensor, double *result, int64_t n);
bool asech_fp32_invoke(float *tensor, float *result, int64_t n);
bool asech_fp64_invoke(double *tensor, double *result, int64_t n);
bool acoth_fp32_invoke(float *tensor, float *result, int64_t n);
bool acoth_fp64_invoke(double *tensor, double *result, int64_t n);
bool acsch2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acsch2_fp64_invoke(double *tensor, double *result, int64_t n);
bool asech2_fp32_invoke(float *tensor, float *result, int64_t n);
bool asech2_fp64_invoke(double *tensor, double *result, int64_t n);
bool acoth2_fp32_invoke(float *tensor, float *result, int64_t n);
bool acoth2_fp64_invoke(double *tensor, double *result, int64_t n);


#endif //SCI_NET_NATIVE_TRIGONOMETRY_KERNELS_H
