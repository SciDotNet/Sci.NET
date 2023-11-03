//
// Created by reece on 02/11/2023.
//

#include "gtest/gtest.h"
#include "mathematics/random.h"
#include "mathematics/memory_api.h"

TEST(random, random_uniform_fp32) {
    const float MIN = 0.0f;
    const float MAX = 1.0f;
    const size_t ELEMENTS = 100;

    float *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(float) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_fp32(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<float *>(calloc(ELEMENTS, sizeof(float)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(float) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_fp64) {
    const float MIN = 0.0;
    const float MAX = 1.0;
    const size_t ELEMENTS = 100;

    double *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(double) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_fp64(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<double *>(calloc(ELEMENTS, sizeof(double)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(double) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}


TEST(random, random_uniform_u8) {
    const uint8_t MIN = 0;
    const uint8_t MAX = 100;
    const size_t ELEMENTS = 100;

    uint8_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(uint8_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_u8(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<uint8_t *>(calloc(ELEMENTS, sizeof(uint8_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(uint8_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}


TEST(random, random_uniform_u16) {
    const uint16_t MIN = 0;
    const uint16_t MAX = 1000;
    const size_t ELEMENTS = 100;

    uint16_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(uint16_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_u16(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<uint16_t *>(calloc(ELEMENTS, sizeof(uint16_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(uint16_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_u32) {
    const uint32_t MIN = 0;
    const uint32_t MAX = 10000;
    const size_t ELEMENTS = 100;

    uint32_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(uint32_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_u32(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<uint32_t *>(calloc(ELEMENTS, sizeof(uint32_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(uint32_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_u64) {
    const uint64_t MIN = 0;
    const uint64_t MAX = 100;
    const size_t ELEMENTS = 100000;

    uint64_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(uint64_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_u64(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<uint64_t *>(calloc(ELEMENTS, sizeof(uint64_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(uint64_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}


TEST(random, random_uniform_i8) {
    const int8_t MIN = -50;
    const int8_t MAX = 50;
    const size_t ELEMENTS = 100;

    int8_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(int8_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_i8(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<int8_t *>(calloc(ELEMENTS, sizeof(int8_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(int8_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_i16) {
    const int16_t MIN = -500;
    const int16_t MAX = 500;
    const size_t ELEMENTS = 100;

    int16_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(int16_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_i16(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<int16_t *>(calloc(ELEMENTS, sizeof(int16_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(int16_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_i32) {
    const int32_t MIN = -5000;
    const int32_t MAX = 5000;
    const size_t ELEMENTS = 100;

    int32_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(int32_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_i32(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<int32_t *>(calloc(ELEMENTS, sizeof(int32_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(int32_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}

TEST(random, random_uniform_i64) {
    const int64_t MIN = -50000;
    const int64_t MAX = 50000;
    const size_t ELEMENTS = 100;

    int64_t *memory;
    auto allocResult = allocate_memory(reinterpret_cast<void **>(&memory), (size_t) sizeof(int64_t) * ELEMENTS);
    ASSERT_EQ(allocResult, sdnSuccess);

    auto randomResult = random_uniform_i64(memory, MIN, MAX, ELEMENTS, 0);
    ASSERT_EQ(randomResult, sdnSuccess);

    auto *copiedMemory = static_cast<int64_t *>(calloc(ELEMENTS, sizeof(int64_t)));
    auto copyResult = copy_memory_to_host(reinterpret_cast<void *>(copiedMemory), reinterpret_cast<void *>(memory),
                                          (size_t) sizeof(int64_t) * ELEMENTS);
    ASSERT_EQ(copyResult, sdnSuccess);

    for (int i = 0; i < ELEMENTS; ++i) {
        ASSERT_GE(copiedMemory[i], MIN);
        ASSERT_LE(copiedMemory[i], MAX);
    }
}