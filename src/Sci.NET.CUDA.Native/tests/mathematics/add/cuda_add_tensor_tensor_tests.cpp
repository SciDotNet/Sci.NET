//
// Created by reece on 04/11/2023.
//

#include "gtest/gtest.h"
#include "mathematics/arithmetic.h"
#include "mathematics/memory_api.h"

TEST(arithmetic_add, add_tensor_tensor_fp32) {

    const int element_count = 8;

    float left_tensor[2][4] = {{1, 2, 3, 4},
                               {5, 6, 7, 8}};

    float right_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    float result_tensor[2][4] = {{0, 0, 0, 0},
                                 {0, 0, 0, 0}};

    float expected[2][4] = {{2,  4,  6,  8},
                            {10, 12, 14, 16}};

    float *left_tensor_d;
    float *right_tensor_d;
    float *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(float) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(float) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(float) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(float) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(float) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_fp32(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(float) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}

TEST(arithmetic_add, add_tensor_tensor_fp64) {

    const int element_count = 8;

    double left_tensor[2][4] = {{1, 2, 3, 4},
                               {5, 6, 7, 8}};

    double right_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    double result_tensor[2][4] = {{0, 0, 0, 0},
                                 {0, 0, 0, 0}};

    double expected[2][4] = {{2,  4,  6,  8},
                            {10, 12, 14, 16}};

    double *left_tensor_d;
    double *right_tensor_d;
    double *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(double) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(double) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(double) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(double) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(double) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_fp64(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(double) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}


TEST(arithmetic_add, add_tensor_tensor_u8) {

    const int element_count = 8;

    uint8_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    uint8_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    uint8_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    uint8_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    uint8_t *left_tensor_d;
    uint8_t *right_tensor_d;
    uint8_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_u8(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(uint8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}


TEST(arithmetic_add, add_tensor_tensor_u16) {

    const int element_count = 8;

    uint16_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    uint16_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    uint16_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    uint16_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    uint16_t *left_tensor_d;
    uint16_t *right_tensor_d;
    uint16_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_u16(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(uint16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}


TEST(arithmetic_add, add_tensor_tensor_u32) {

    const int element_count = 8;

    uint32_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    uint32_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    uint32_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    uint32_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    uint32_t *left_tensor_d;
    uint32_t *right_tensor_d;
    uint32_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_u32(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(uint32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}

TEST(arithmetic_add, add_tensor_tensor_u64) {

    const int element_count = 8;

    uint64_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    uint64_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    uint64_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    uint64_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    uint64_t *left_tensor_d;
    uint64_t *right_tensor_d;
    uint64_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_u64(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(uint64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}


TEST(arithmetic_add, add_tensor_tensor_i8) {

    const int element_count = 8;

    int8_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    int8_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    int8_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    int8_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    int8_t *left_tensor_d;
    int8_t *right_tensor_d;
    int8_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_i8(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(int8_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}

TEST(arithmetic_add, add_tensor_tensor_i16) {

    const int element_count = 8;

    int16_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    int16_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    int16_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    int16_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    int16_t *left_tensor_d;
    int16_t *right_tensor_d;
    int16_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_i16(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(int16_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}


TEST(arithmetic_add, add_tensor_tensor_i32) {

    const int element_count = 8;

    int32_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    int32_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    int32_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    int32_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    int32_t *left_tensor_d;
    int32_t *right_tensor_d;
    int32_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_i32(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(int32_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}

TEST(arithmetic_add, add_tensor_tensor_i64) {

    const int element_count = 8;

    int64_t left_tensor[2][4] = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};

    int64_t right_tensor[2][4] = {{1, 2, 3, 4},
                                 {5, 6, 7, 8}};

    int64_t result_tensor[2][4] = {{0, 0, 0, 0},
                                  {0, 0, 0, 0}};

    int64_t expected[2][4] = {{2,  4,  6,  8},
                             {10, 12, 14, 16}};

    int64_t *left_tensor_d;
    int64_t *right_tensor_d;
    int64_t *result_tensor_d;

    auto allocResult = allocate_memory(reinterpret_cast<void **>(&left_tensor_d),
                                       (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&right_tensor_d), (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    allocResult = allocate_memory(reinterpret_cast<void **>(&result_tensor_d), (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto copyResult = copy_memory_to_device(reinterpret_cast<void *>(left_tensor_d),
                                            reinterpret_cast<void *>(left_tensor),
                                            (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    copyResult = copy_memory_to_device(reinterpret_cast<void *>(right_tensor_d), reinterpret_cast<void *>(right_tensor),
                                       (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    auto randomResult = add_tensor_tensor_i64(left_tensor_d, right_tensor_d, result_tensor_d, element_count);
    ASSERT_EQ(randomResult, sdnSuccess);

    copyResult = copy_memory_to_host(reinterpret_cast<void *>(result_tensor), reinterpret_cast<void *>(result_tensor_d),
                                     (size_t) sizeof(int64_t) * element_count);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(result_tensor[i][j], expected[i][j]);
        }
    }
}