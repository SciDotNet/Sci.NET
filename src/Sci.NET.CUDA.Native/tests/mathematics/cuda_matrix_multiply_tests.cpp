//
// Created by reece on 03/11/2023.
//

#include "gtest/gtest.h"
#include "mathematics/matrix_multiply.h"
#include "mathematics/memory_api.h"

#define BF16_1 0x3f80
#define BF16_2 0x4000
#define BF16_3 0x4040
#define BF16_4 0x4080
#define BF16_30 0x41f0

TEST(matrix_multiply, random_uniform_bf16) {

    bfloat16 matrixA[2][4] = {{BF16_1, BF16_2, BF16_3, BF16_4},
                              {BF16_1, BF16_2, BF16_3, BF16_4}};

    bfloat16 matrixB[4][2] = {{BF16_1, BF16_1},
                              {BF16_2, BF16_2},
                              {BF16_3, BF16_3},
                              {BF16_4, BF16_4}};

    bfloat16 matrixC[2][2] = {{0, 0},
                              {0, 0}};

    bfloat16 expected[2][2] = {{BF16_30, BF16_30},
                               {BF16_30, BF16_30}};

    bfloat16 *matrixA_d;
    bfloat16 *matrixB_d;
    bfloat16 *matrixC_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&matrixA_d), (size_t) sizeof(bfloat16) * 2 * 4);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&matrixB_d), (size_t) sizeof(bfloat16) * 4 * 2);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&matrixC_d), (size_t) sizeof(bfloat16) * 2 * 2);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(matrixA_d), reinterpret_cast<void *>(matrixA),
                                            (size_t) sizeof(bfloat16) * 2 * 4);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(matrixB_d), reinterpret_cast<void *>(matrixB),
                                       (size_t) sizeof(bfloat16) * 4 * 2);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = matrix_multiply_bf16(matrixA_d, matrixB_d, matrixC_d, 2, 4, 2);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(matrixC), reinterpret_cast<void *>(matrixC_d),
                                     (size_t) sizeof(bfloat16) * 2 * 2);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(matrixC[i][j], expected[i][j]);
        }
    }
}


TEST(matrix_multiply, random_uniform_fp32) {

    float matrixA[2][4] = {{1, 2, 3, 4},
                           {1, 2, 3, 4}};

    float matrixB[4][2] = {{1, 1},
                           {2, 2},
                           {3, 3},
                           {4, 4}};

    float matrixC[2][2] = {{0, 0},
                           {0, 0}};

    float expected[2][2] = {{30, 30},
                            {30, 30}};

    float *matrixA_d;
    float *matrixB_d;
    float *matrixC_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&matrixA_d), (size_t) sizeof(float) * 2 * 4);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&matrixB_d), (size_t) sizeof(float) * 4 * 2);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&matrixC_d), (size_t) sizeof(float) * 2 * 2);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(matrixA_d), reinterpret_cast<void *>(matrixA),
                                            (size_t) sizeof(float) * 2 * 4);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(matrixB_d), reinterpret_cast<void *>(matrixB),
                                       (size_t) sizeof(float) * 4 * 2);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = matrix_multiply_fp32(matrixA_d, matrixB_d, matrixC_d, 2, 4, 2);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(matrixC), reinterpret_cast<void *>(matrixC_d),
                                     (size_t) sizeof(float) * 2 * 2);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(matrixC[i][j], expected[i][j]);
        }
    }
}
