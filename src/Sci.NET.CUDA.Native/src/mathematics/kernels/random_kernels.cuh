//
// Created by reece on 02/08/2023.
//

#ifndef SCI_NET_NATIVE_RANDOM_KERNELS_CUH
#define SCI_NET_NATIVE_RANDOM_KERNELS_CUH

bool random_uniform_fp64_invoke(double *dst,
                                double min,
                                double max,
                                size_t n,
                                long seed);

bool random_uniform_fp32_invoke(float *dst,
                                float min,
                                float max,
                                size_t n,
                                long seed);

bool random_uniform_u8_invoke(uint8_t *dst,
                              uint8_t min,
                              uint8_t max,
                              size_t n,
                              long seed);

bool random_uniform_u16_invoke(uint16_t *dst,
                               uint16_t min,
                               uint16_t max,
                               size_t n,
                               long seed);

bool random_uniform_u32_invoke(uint32_t *dst,
                               uint32_t min,
                               uint32_t max,
                               size_t n,
                               long seed);

bool random_uniform_u64_invoke(uint64_t *dst,
                               uint64_t min,
                               uint64_t max,
                               size_t n,
                               long seed);

bool random_uniform_i8_invoke(int8_t *dst,
                              int8_t min,
                              int8_t max,
                              size_t n,
                              long seed);

bool random_uniform_i16_invoke(int16_t *dst,
                               int16_t min,
                               int16_t max,
                               size_t n,
                               long seed);

bool random_uniform_i32_invoke(int32_t *dst,
                               int32_t min,
                               int32_t max,
                               size_t n,
                               long seed);

bool random_uniform_i64_invoke(int64_t *dst,
                               int64_t min,
                               int64_t max,
                               size_t n,
                               long seed);


#endif //SCI_NET_NATIVE_RANDOM_KERNELS_CUH
