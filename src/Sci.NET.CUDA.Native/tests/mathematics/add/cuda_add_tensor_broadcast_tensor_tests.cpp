//
// Created by reece on 04/11/2023.
//

#include "gtest/gtest.h"
#include "mathematics/arithmetic.h"
#include "mathematics/memory_api.h"

TEST(arithmetic, add_tensor_broadcast_tensor_fp32) {

    float left_tensor[4] = {1, 2, 3, 4};

    float right_tensor[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};

    float result_tensor[2][4] = {{0, 0, 0, 0},
                                 {0, 0, 0, 0}};

    float expected[2][4] = {{2, 4, 6, 8},
                            {6, 8, 10, 12}};

    float *left_tensor_d;
    float *right_tensor_d;
    float *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d), (size_t) sizeof(float) * 4);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(float) * 2 * 4);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(float) * 2 * 4);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(float) * 1);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(float) * 4 * 2);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_broadcast_tensor_fp32(left_tensor_d, right_tensor_d, result_tensor_d, 2, 4);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(float) * 2 * 4);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}