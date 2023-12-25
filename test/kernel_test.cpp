#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "ImgStatics.hpp"
#include "ImgTransform.hpp"
#include "OclInfo.hpp"
#include "ScalarBuffer.hpp"
#include "random_case_generator.hpp"

using namespace fingerprint_parallel::core;

#define PI 3.141592

TEST(ImageTransformTest, GrayScale) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);
    ImgStatics img_statics(ocl_info);

    //  0: width, 1: height, 2: original data (shape.xy=(3*width,height)),
    // 3: expected result
    using grayscale_datatype =
        std::tuple<int, int, std::vector<uint8_t>, std::vector<uint8_t>>;

    std::vector<grayscale_datatype> datasets{
        {3,
         3,
         {119, 64,  92,  35,  231, 56,  38,  101, 69,   // [R G B R G B R G B]
          229, 210, 2,   249, 59,  32,  175, 254, 107,  //
          85,  173, 184, 231, 236, 255, 96,  166, 14},
         {105, 77, 53, 209, 193, 186, 110, 233, 104}},
        {3,
         3,
         {0,   49,  34,  48,  25,  246, 57,  75,  166,  // [R G B R G B R G B]
          141, 179, 10,  55,  17,  250, 141, 118, 173,  //
          202, 39,  232, 216, 138, 33,  217, 197, 244},
         {12, 57, 68, 139, 60, 138, 169, 186, 214}},
    };

    // create random data
    RandomMatrixGenerator generator;

    std::mt19937_64 gen(47);
    std::uniform_int_distribution<int> width_dis(3, 300);

    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        const int width = width_dis(gen);
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, width * 3, 10);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);
        std::vector<uint8_t> result(width * std::get<1>(input_data));

        for (int i = 0; i < result.size(); ++i) {
            int v = arr[3 * i + 0] * 0.72f + arr[3 * i + 1] * 0.21f +
                    arr[3 * i + 2] * 0.07f;

            result[i] = v;
        }

        datasets.push_back({width, std::get<1>(input_data), arr, result});
    }

    auto test_one_pair = [&](grayscale_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data) * 3, std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<0>(data),
                                            std::get<1>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        Img img(buffer_original, Img::RGB);

        cl::ImageFormat img_format(CL_RGBA, CL_UNSIGNED_INT8);
        cl::Image2D climg(ocl_info.ctx_, CL_MEM_READ_WRITE, img_format,
                          img.width(), img.height(), 0, 0);
        int err = ocl_info.queue_.enqueueWriteImage(
            climg, CL_FALSE, {0, 0, 0}, {img.width(), img.height(), 1}, 0, 0,
            img.data());
        if (err) throw OclException("Error while enqueue image", err);

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        img_transformer.to_gray_scale(climg, buffer_result);

        buffer_result.to_host();

        for (int i = 0; i < std::get<2>(data).size(); ++i) {
            const int v1 = buffer_result.data()[i];
            const int v2 = buffer_result.data()[i];
            const int diff = v1 - v2;
            ASSERT_TRUE(diff < 2 && diff > -2);
        }
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, Negate) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);

    std::vector<uint8_t> vOriginal(256);  // 0,1,2, ... ,255
    std::vector<uint8_t> vExpected(256);  // 255,254, ... , 0

    for (int i = 0; i < 256; ++i) {
        vOriginal[i] = i;
        vExpected[i] = 255 - i;
    }

    // 0: width, 1: height, 2: original data, 3: expected result
    using negate_datatype =
        std::tuple<int, int, std::vector<uint8_t>, std::vector<uint8_t>>;

    std::vector<negate_datatype> datasets{
        {1, 256, vOriginal, vExpected},
        {3,
         3,
         {0, 127, 255, 1, 128, 254, 2, 129, 253},
         {255, 128, 0, 254, 127, 1, 253, 126, 2}},
        {1, 1, {0}, {255}},
        {1, 1, {255}, {0}}};

    // create random data
    RandomMatrixGenerator generator;

    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, 16, 512);

        const std::vector<uint8_t>& arr = std::get<2>(input_data);
        const int N = arr.size();

        std::vector<uint8_t> expected(N);

        for (int i = 0; i < arr.size(); ++i) {
            expected[i] = 255 - arr[i];
        }

        datasets.push_back(
            {std::get<0>(input_data), std::get<1>(input_data), arr, expected});
    }

    auto test_one_pair = [&](negate_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<0>(data),
                                            std::get<1>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        img_transformer.negate(buffer_original, buffer_result);

        buffer_result.to_host();

        ASSERT_EQ(buffer_result, buffer_expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, Copy) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);

    MatrixBuffer<uint8_t> buffer_original({1, 50, 126, 200, 255});
    MatrixBuffer<uint8_t> buffer_copied(1, 5);

    buffer_original.create_buffer(&ocl_info);
    buffer_copied.create_buffer(&ocl_info);

    buffer_original.to_gpu();

    img_transformer.copy(buffer_original, buffer_copied);
    buffer_copied.to_host();

    ASSERT_EQ(buffer_copied, buffer_original);
}

TEST(ImageTransformTest, Normalize) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);
    ImgStatics img_statics(ocl_info);

    // 0: M0, 1: V0, 2: width, 3: height, 4: original data, 5: expected result
    using normalize_datatype =
        std::tuple<int, int, int, int, std::vector<uint8_t>,
                   std::vector<uint8_t>>;

    std::vector<normalize_datatype> datasets{
        {128,
         2000,
         3,
         3,
         {237, 163, 52, 65, 129, 218, 62, 148, 212},
         {190, 141, 67, 76, 118, 177, 74, 131, 173}},
        {128,
         2000,
         3,
         3,
         {74, 38, 195, 187, 41, 68, 86, 117, 114},
         {104, 74, 204, 198, 77, 99, 114, 140, 137}},
    };

    // create random data
    RandomMatrixGenerator generator;

    std::mt19937_64 gen(47);
    std::uniform_int_distribution<int> mean_dis(0, 255);
    std::uniform_int_distribution<int> var_dis(0, 5000);

    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, 5, 32);

        const int NC = std::get<0>(input_data);
        const int NR = std::get<1>(input_data);
        const std::vector<uint8_t>& arr = std::get<2>(input_data);

        const float mean0 = static_cast<float>(mean_dis(gen));
        const float var0 = static_cast<float>(var_dis(gen));
        int64_t sum = 0;
        int64_t square_sum = 0;
        const int N = arr.size();

        for (int64_t value : arr) {
            sum += value;
            square_sum += value * value;
        }

        float mean = static_cast<float>(sum) / N;
        float var = static_cast<float>(square_sum) / N - mean * mean;

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            float pixel = static_cast<float>(arr[i]);
            float delta = abs(pixel - mean) * sqrtf(var0 / var);
            int val = pixel > mean ? mean0 + delta : mean0 - delta;
            result[i] = std::clamp(val, 0, 255);
        }

        datasets.push_back({mean0, var0, NC, NR, arr, result});
    }

    auto test_one_pair = [&](normalize_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<2>(data), std::get<3>(data), std::get<4>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<2>(data),
                                            std::get<3>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<2>(data), std::get<3>(data), std::get<5>(data));

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        ScalarBuffer<float> mean, var;

        mean.create_buffer(&ocl_info);
        var.create_buffer(&ocl_info);

        img_statics.mean(buffer_original, mean);
        img_statics.var(buffer_original, var);

        img_transformer.normalize(buffer_original, buffer_result,
                                  std::get<0>(data), std::get<1>(data), mean,
                                  var);

        buffer_result.to_host();

        ASSERT_EQ(buffer_result, buffer_expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, DynamicThresholding) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);

    // 0: block_size, 1: scale, 2: width, 3: height, 4: original data, 5:
    // expected result
    using dynamic_thresholding_datatype =
        std::tuple<int, float, int, int, std::vector<uint8_t>,
                   std::vector<uint8_t>>;

    std::vector<dynamic_thresholding_datatype> datasets{
        {
            3,
            1.05,
            7,
            5,
            {0, 0,   0,   0,   0,   0,   0,  //
             0, 100, 100, 100, 100, 100, 0,  //
             0, 100, 100, 100, 100, 100, 0,  //
             0, 100, 100, 100, 100, 100, 0,  //
             0, 0,   0,   0,   0,   0,   0},
            {0, 0,   0,   0,   0,   0,   0,  //
             0, 255, 255, 255, 255, 255, 0,  //
             0, 255, 0,   0,   0,   255, 0,  //
             0, 255, 255, 255, 255, 255, 0,  //
             0, 0,   0,   0,   0,   0,   0}  //
        },
        {
            5,
            1.05,
            9,
            7,
            {
                0, 0, 0,   0,   0,   0,   0,   0, 0,  //
                0, 0, 0,   0,   0,   0,   0,   0, 0,  //
                0, 0, 100, 100, 100, 100, 100, 0, 0,  //
                0, 0, 100, 100, 100, 100, 100, 0, 0,  //
                0, 0, 100, 100, 100, 100, 100, 0, 0,  //
                0, 0, 0,   0,   0,   0,   0,   0, 0,  //
                0, 0, 0,   0,   0,   0,   0,   0, 0,
            },
            {0, 0, 0,   0,   0,   0,   0,   0, 0,  //
             0, 0, 0,   0,   0,   0,   0,   0, 0,  //
             0, 0, 255, 255, 255, 255, 255, 0, 0,  //
             0, 0, 255, 255, 255, 255, 255, 0, 0,  //
             0, 0, 255, 255, 255, 255, 255, 0, 0,  //
             0, 0, 0,   0,   0,   0,   0,   0, 0,  //
             0, 0, 0,   0,   0,   0,   0,   0, 0}  //
        },
    };

    // create random data
    RandomMatrixGenerator generator;

    std::mt19937_64 gen(47);
    std::uniform_int_distribution<int> halfblock_size_dist(1, 3);
    std::uniform_real_distribution<float> scale_dis(0.8, 1.2);

    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255);

        const int NC = std::get<0>(input_data);
        const int NR = std::get<1>(input_data);
        const std::vector<uint8_t>& arr = std::get<2>(input_data);

        const int halfblock_size = halfblock_size_dist(gen);
        const int block_size = halfblock_size * 2 + 1;
        const float scale = scale_dis(gen);

        const auto value = [&](int r, int c) -> const uint8_t {
            if (r < 0 || r >= NR || c < 0 || c >= NC) {
                return 0;
            }

            return arr[NC * r + c];
        };

        const auto dynamicThresholdVal = [&](int idx) -> const uint8_t {
            const int r = idx / NC;
            const int c = idx % NC;

            float sum = 0;

            for (int nextR = r - halfblock_size; nextR <= r + halfblock_size;
                 ++nextR) {
                for (int nextC = c - halfblock_size;
                     nextC <= c + halfblock_size; ++nextC) {
                    sum += value(nextR, nextC);
                }
            }

            float mean = sum / ((block_size * block_size));
            mean *= scale;

            return value(r, c) > mean ? 255 : 0;
        };

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            result[i] = dynamicThresholdVal(i);
        }

        datasets.push_back({block_size, scale, NC, NR, arr, result});
    }

    auto test_one_pair = [&](dynamic_thresholding_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<2>(data), std::get<3>(data), std::get<4>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<2>(data),
                                            std::get<3>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<2>(data), std::get<3>(data), std::get<5>(data));

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        img_transformer.dynamic_thresholding(buffer_original, buffer_result,
                                             std::get<0>(data),
                                             std::get<1>(data));

        buffer_result.to_host();

        ASSERT_EQ(buffer_result, buffer_expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, ApplyGaussian) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using gaussian_datatype =
        std::tuple<int, int, std::vector<uint8_t>, std::vector<uint8_t>>;

    std::vector<gaussian_datatype> datasets{
        {5,
         5,
         {0, 0, 0,   0, 0,  //
          0, 0, 0,   0, 0,  //
          0, 0, 100, 0, 0,  //
          0, 0, 0,   0, 0,  //
          0, 0, 0,   0, 0},
         {0, 0,  0,  0,  0,  //
          0, 6,  13, 6,  0,  //
          0, 13, 25, 13, 0,  //
          0, 6,  13, 6,  0,  //
          0, 0,  0,  0,  0}},
        {5,
         5,
         {0,   0,   100, 0,   0,    //
          0,   0,   100, 0,   0,    //
          100, 100, 100, 100, 100,  //
          0,   0,   100, 0,   0,    //
          0,   0,   100, 0,   0},
         {0,  19, 38, 19, 0,   //
          19, 44, 63, 44, 19,  //
          38, 63, 75, 63, 38,  //
          19, 44, 63, 44, 19,  //
          0,  19, 38, 19, 0}},
    };

    // create random data
    RandomMatrixGenerator generator;
    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, 5, 5);

        const int NC = std::get<0>(input_data);
        const int NR = std::get<1>(input_data);
        const std::vector<uint8_t>& arr = std::get<2>(input_data);

        const auto value = [&](int r, int c) -> const uint8_t {
            if (r < 0 || r >= NR || c < 0 || c >= NC) {
                return 0;
            }

            return arr[NC * r + c];
        };

        const int dx[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        const int dy[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
        const int weight[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

        const auto gaussianVal = [&](int idx) -> const uint8_t {
            const int r = idx / NC;
            const int c = idx % NC;

            uint8_t ret = 0;
            int sum = 0;

            for (int i = 0; i < 9; ++i) {
                sum += value(r + dx[i], c + dy[i]) * weight[i];
            }

            ret = (sum + 8) / 16;

            return ret;
        };

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            result[i] = gaussianVal(i);
        }

        datasets.push_back({NC, NR, arr, result});
    }

    auto test_one_pair = [&](gaussian_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<0>(data),
                                            std::get<1>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        img_transformer.gaussian_filter(buffer_original, buffer_result);

        buffer_result.to_host();

        ASSERT_EQ(buffer_result, buffer_expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, Rotate) {
    OclInfo ocl_info = OclInfo::init_opencl();
    ImgTransform img_transformer(ocl_info);

    //  0: width, 1: height, 2: original data, 3: expected result
    using rotate_datatype =
        std::tuple<int, int, std::vector<uint8_t>, std::vector<uint8_t>, float>;

    std::vector<rotate_datatype> datasets;

    // create random data
    RandomMatrixGenerator generator;
    std::mt19937_64 gen(47);
    std::uniform_real_distribution<float> degree_dis(-1.0f * PI, 1.0f * PI);

    const int n_random_cases = 100;
    for (int random_case_no = 0; random_case_no < n_random_cases;
         ++random_case_no) {
        std::tuple<int, int, std::vector<uint8_t>> input_data =
            generator.generate_matrix_data(0, 255, 11, 100);

        const int NC = std::get<0>(input_data);
        const int NR = std::get<1>(input_data);
        const std::vector<uint8_t>& arr = std::get<2>(input_data);

        const int kCenterX = NC / 2;
        const int kCenterY = NR / 2;

        const float degree = degree_dis(gen);

        const auto value = [&](int r, int c) -> const uint8_t {
            if (r < 0 || r >= NR || c < 0 || c >= NC) {
                return 0;
            }

            return arr[NC * r + c];
        };

        const auto rotateVal = [&](int idx) -> const uint8_t {
            const int r = idx / NC;
            const int c = idx % NC;

            const float dx = c - kCenterX;
            const float dy = r - kCenterY;

            const float sin_val = std::sin(-degree);
            const float cos_val = std::cos(-degree);

            int target_x = cos_val * dx - sin_val * dy + 0.5 + kCenterX;
            int target_y = sin_val * dx + cos_val * dy + 0.5 + kCenterY;

            return value(target_y, target_x);
        };

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            result[i] = rotateVal(i);
        }

        datasets.push_back({NC, NR, arr, result, degree});
    }

    auto test_one_pair = [&](rotate_datatype& data) {
        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> buffer_result(std::get<0>(data),
                                            std::get<1>(data));
        MatrixBuffer<uint8_t> buffer_expected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));
        const float degree = std::get<4>(data);

        buffer_original.create_buffer(&ocl_info);
        buffer_result.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        img_transformer.rotate(buffer_original, buffer_result, degree);

        buffer_result.to_host();

        ASSERT_EQ(buffer_result, buffer_expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}