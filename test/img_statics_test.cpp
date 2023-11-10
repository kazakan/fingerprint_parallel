#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "ImgStatics.hpp"
#include "OclInfo.hpp"
#include "ScalarBuffer.hpp"
#include "random_case_generator.hpp"

using namespace fingerprint_parallel::core;

TEST(ImgStaticsTest, Sum) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgStatics img_statics(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, std::vector<uint8_t>, int>;

    std::vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 45},
        {3, 3, {255, 255, 255, 255, 255, 255, 255, 255, 255}, 2295},
        {3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 0},
        {1, 1, {1}, 1},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);
        const int N = arr.size();

        int64_t sum = 0;

        for (int i = 0; i < N; ++i) {
            sum += arr[i];
        }

        datasets.push_back(
            {std::get<0>(input_data), std::get<1>(input_data), arr, sum});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        double expected = std::get<3>(data);

        ScalarBuffer<uint64_t> result;

        buffer_original.create_buffer(ocl_info.ctx_);
        buffer_original.to_gpu(ocl_info);
        result.create_buffer(ocl_info.ctx_);

        img_statics.sum(buffer_original, result);
        result.to_host(ocl_info);

        ASSERT_EQ(result.value(), expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, SqaureSum) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgStatics img_statics(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, std::vector<uint8_t>, int64_t>;

    std::vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 285},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);
        const int N = arr.size();

        int64_t sum = 0;

        for (int64_t v : arr) {
            sum += v * v;
        }

        datasets.push_back(
            {std::get<0>(input_data), std::get<1>(input_data), arr, sum});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        ScalarBuffer<uint64_t> result;
        double expected = std::get<3>(data);

        buffer_original.create_buffer(ocl_info.ctx_);
        buffer_original.to_gpu(ocl_info);
        result.create_buffer(ocl_info.ctx_);

        img_statics.square_sum(buffer_original, result);

        result.to_host(ocl_info);

        ASSERT_EQ(result.value(), expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, Mean) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgStatics img_statics(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, std::vector<uint8_t>, float>;

    std::vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 5},
        {3, 3, {255, 255, 255, 255, 255, 255, 255, 255, 255}, 255},
        {3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 0},
        {1, 1, {1}, 1},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);
        const int N = arr.size();

        int64_t sum = 0;

        for (int i = 0; i < N; ++i) {
            sum += arr[i];
        }

        float mean = static_cast<float>(sum) / N;

        datasets.push_back(
            {std::get<0>(input_data), std::get<1>(input_data), arr, mean});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        ScalarBuffer<float> result;
        float expected = std::get<3>(data);

        buffer_original.create_buffer(ocl_info.ctx_);
        buffer_original.to_gpu(ocl_info);

        result.create_buffer(ocl_info.ctx_);

        img_statics.mean(buffer_original, result);

        result.to_host(ocl_info);

        ASSERT_NEAR(result.value(), expected, 0.0001);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, Var) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgStatics img_statics(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using var_datatype = std::tuple<int, int, std::vector<uint8_t>, float>;

    std::vector<var_datatype> datasets{
        {3, 3, {76, 49, 136, 167, 143, 160, 75, 220, 71}, 2884.98765432},
        {3, 3, {102, 174, 55, 135, 45, 115, 40, 216, 40}, 3620.24691358024}};

    // Create random data
    RandomMatrixGenerator generator;
    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, 16, 512);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);

        int64_t sum = 0;
        int64_t square_sum = 0;
        const int N = arr.size();

        for (int i = 0; i < N; ++i) {
            int64_t v = arr[i];
            sum += v;
            square_sum += v * v;
        }

        float mean = static_cast<float>(sum) / N;
        float expected = static_cast<float>(square_sum) / N - mean * mean;

        datasets.push_back(
            {std::get<0>(input_data), std::get<1>(input_data), arr, expected});
    }

    auto test_one_pair = [&](var_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        ScalarBuffer<float> result;

        float expected = std::get<3>(data);

        buffer_original.create_buffer(ocl_info.ctx_);
        buffer_original.to_gpu(ocl_info);

        result.create_buffer(ocl_info.ctx_);

        img_statics.var(buffer_original, result);
        result.to_host(ocl_info);

        float relative_err = abs((result.value() - expected) / result.value());

        ASSERT_LE(relative_err, 0.000001);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}