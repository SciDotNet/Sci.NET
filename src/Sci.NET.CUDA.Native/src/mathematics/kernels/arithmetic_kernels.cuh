//
// Created by reece on 02/08/2023.
//

#ifndef SCI_NET_NATIVE_ARITHMETIC_KERNELS_CUH
#define SCI_NET_NATIVE_ARITHMETIC_KERNELS_CUH


bool add_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t n);

bool add_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t n);

bool add_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);

bool add_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t n);

bool add_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);

bool add_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t n);

bool add_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);

bool add_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t n);

bool add_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);

bool add_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t n);

bool add_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);

bool add_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);

bool add_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool subtract_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t n);

bool subtract_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t n);

bool subtract_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);

bool subtract_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t n);

bool subtract_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);

bool subtract_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t n);

bool subtract_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);

bool subtract_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t n);

bool subtract_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);

bool subtract_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t n);

bool subtract_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);

bool subtract_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);

bool subtract_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool multiply_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t n);
bool multiply_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t n);
bool multiply_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
bool multiply_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t n);
bool multiply_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
bool multiply_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t n);
bool multiply_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
bool multiply_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t n);
bool multiply_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
bool multiply_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t n);
bool multiply_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
bool multiply_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
bool multiply_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool divide_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t n);
bool divide_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t n);
bool divide_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
bool divide_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t n);
bool divide_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
bool divide_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t n);
bool divide_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
bool divide_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t n);
bool divide_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
bool divide_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t n);
bool divide_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
bool divide_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
bool divide_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

bool negate_tensor_fp32_invoke(float *a, float *result, int64_t n);
bool negate_tensor_fp64_invoke(double *a, double *result, int64_t n);
bool negate_tensor_i8_invoke(int8_t *a, int8_t *result, int64_t n);
bool negate_tensor_i16_invoke(int16_t *a, int16_t *result, int64_t n);
bool negate_tensor_i32_invoke(int32_t *a, int32_t *result, int64_t n);
bool negate_tensor_i64_invoke(int64_t *a, int64_t *result, int64_t n);

bool abs_tensor_fp32_invoke(float *a, float *result, int64_t n);
bool abs_tensor_fp64_invoke(double *a, double *result, int64_t n);
bool abs_tensor_i8_invoke(int8_t *a, int8_t *result, int64_t n);
bool abs_tensor_i16_invoke(int16_t *a, int16_t *result, int64_t n);
bool abs_tensor_i32_invoke(int32_t *a, int32_t *result, int64_t n);
bool abs_tensor_i64_invoke(int64_t *a, int64_t *result, int64_t n);

bool sqrt_tensor_fp32_invoke(float *a, float *result, int64_t n);
bool sqrt_tensor_fp64_invoke(double *a, double *result, int64_t n);
bool sqrt_tensor_u8_invoke(uint8_t *a, uint8_t *result, int64_t n);
bool sqrt_tensor_i8_invoke(int8_t *a, int8_t *result, int64_t n);
bool sqrt_tensor_u16_invoke(uint16_t *a, uint16_t *result, int64_t n);
bool sqrt_tensor_i16_invoke(int16_t *a, int16_t *result, int64_t n);
bool sqrt_tensor_u32_invoke(uint32_t *a, uint32_t *result, int64_t n);
bool sqrt_tensor_i32_invoke(int32_t *a, int32_t *result, int64_t n);
bool sqrt_tensor_u64_invoke(uint64_t *a, uint64_t *result, int64_t n);
bool sqrt_tensor_i64_invoke(int64_t *a, int64_t *result, int64_t n);



#endif //SCI_NET_NATIVE_ARITHMETIC_KERNELS_CUH
