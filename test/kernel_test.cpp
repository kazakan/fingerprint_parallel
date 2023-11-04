#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "ImgStatics.hpp"
#include "ImgTransform.hpp"
#include "OclInfo.hpp"
#include "random_case_generator.hpp"

TEST(ImageTransformTest, GrayScale) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);

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
    std::uniform_int_distribution<int> widthDis(3, 300);

    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        const int width = widthDis(gen);
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 255, width * 3, 10);

        const std::vector<uint8_t>& arr = std::get<2>(inputData);
        std::vector<uint8_t> result(width * std::get<1>(inputData));

        for (int i = 0; i < result.size(); ++i) {
            int v = arr[3 * i + 0] * 0.72f + arr[3 * i + 1] * 0.21f +
                    arr[3 * i + 2] * 0.07f;

            result[i] = v;
        }

        datasets.push_back({width, std::get<1>(inputData), arr, result});
    }

    auto test_one_pair = [&](grayscale_datatype& data) {
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<0>(data) * 3, std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<0>(data),
                                           std::get<1>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        Img img(bufferOriginal, Img::RGB);

        cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
        cl::Image2D climg(oclInfo.ctx, CL_MEM_READ_WRITE, imgFormat, img.width,
                          img.height, 0, 0);
        int err = oclInfo.queue.enqueueWriteImage(climg, CL_FALSE, {0, 0, 0},
                                                  {img.width, img.height, 1}, 0,
                                                  0, img.data);
        if (err) throw OclException("Error while enqueue image", err);

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.toGrayScale(climg, bufferResult);

        bufferResult.toHost(oclInfo);

        for (int i = 0; i < std::get<2>(data).size(); ++i) {
            const int v1 = bufferResult.getData()[i];
            const int v2 = bufferResult.getData()[i];
            const int diff = v1 - v2;
            ASSERT_TRUE(diff < 2 && diff > -2);
        }
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, Negate) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

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

    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 255);

        const std::vector<uint8_t>& arr = std::get<2>(inputData);
        const int N = arr.size();

        std::vector<uint8_t> expected(N);

        for (int i = 0; i < arr.size(); ++i) {
            expected[i] = 255 - arr[i];
        }

        datasets.push_back(
            {std::get<0>(inputData), std::get<1>(inputData), arr, expected});
    }

    auto test_one_pair = [&](negate_datatype& data) {
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<0>(data),
                                           std::get<1>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.negate(bufferOriginal, bufferResult);

        bufferResult.toHost(oclInfo);

        ASSERT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, Copy) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<uint8_t> bufferOriginal({1, 50, 126, 200, 255});
    MatrixBuffer<uint8_t> bufferCopied(1, 5);

    bufferOriginal.createBuffer(oclInfo.ctx);
    bufferCopied.createBuffer(oclInfo.ctx);

    bufferOriginal.toGpu(oclInfo);

    imgTransformer.copy(bufferOriginal, bufferCopied);
    bufferCopied.toHost(oclInfo);

    ASSERT_EQ(bufferCopied, bufferOriginal);
}

TEST(ImageTransformTest, Normalize) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);

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
    std::uniform_int_distribution<int> meanDis(0, 255);
    std::uniform_int_distribution<int> varDis(0, 5000);

    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 255, 5, 5);

        const int NC = std::get<0>(inputData);
        const int NR = std::get<1>(inputData);
        const std::vector<uint8_t>& arr = std::get<2>(inputData);

        const float mean0 = static_cast<float>(meanDis(gen));
        const float var0 = static_cast<float>(varDis(gen));
        long long sum = 0;
        long long squareSum = 0;
        const int N = arr.size();

        for (int value : arr) {
            sum += value;
            squareSum += value * value;
        }

        double mean = static_cast<double>(sum) / N;
        double var = static_cast<double>(squareSum) / N - mean * mean;

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
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<2>(data), std::get<3>(data), std::get<4>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<2>(data),
                                           std::get<3>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<2>(data), std::get<3>(data), std::get<5>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        float mean = imgStatics.mean(bufferOriginal);
        float var = imgStatics.var(bufferOriginal);

        imgTransformer.normalize(bufferOriginal, bufferResult,
                                 std::get<0>(data), std::get<1>(data), mean,
                                 var);

        bufferResult.toHost(oclInfo);

        ASSERT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, DynamicThresholding) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    // 0: blockSize, 1: scale, 2: width, 3: height, 4: original data, 5:
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
    std::uniform_int_distribution<int> halfBlockSizeDis(1, 3);
    std::uniform_real_distribution<float> scaleDis(0.8, 1.2);

    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 255);

        const int NC = std::get<0>(inputData);
        const int NR = std::get<1>(inputData);
        const std::vector<uint8_t>& arr = std::get<2>(inputData);

        const int halfBlockSize = halfBlockSizeDis(gen);
        const int blockSize = halfBlockSize * 2 + 1;
        const float scale = scaleDis(gen);

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

            for (int nextR = r - halfBlockSize; nextR <= r + halfBlockSize;
                 ++nextR) {
                for (int nextC = c - halfBlockSize; nextC <= c + halfBlockSize;
                     ++nextC) {
                    sum += value(nextR, nextC);
                }
            }

            float mean = sum / ((blockSize * blockSize));
            mean *= scale;

            return value(r, c) > mean ? 255 : 0;
        };

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            result[i] = dynamicThresholdVal(i);
        }

        datasets.push_back({blockSize, scale, NC, NR, arr, result});
    }

    auto test_one_pair = [&](dynamic_thresholding_datatype& data) {
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<2>(data), std::get<3>(data), std::get<4>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<2>(data),
                                           std::get<3>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<2>(data), std::get<3>(data), std::get<5>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.applyDynamicThresholding(
            bufferOriginal, bufferResult, std::get<0>(data), std::get<1>(data));

        bufferResult.toHost(oclInfo);

        ASSERT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, ApplyGaussian) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

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
    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 255, 5, 5);

        const int NC = std::get<0>(inputData);
        const int NR = std::get<1>(inputData);
        const std::vector<uint8_t>& arr = std::get<2>(inputData);

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
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<0>(data),
                                           std::get<1>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.applyGaussianFilter(bufferOriginal, bufferResult);

        bufferResult.toHost(oclInfo);

        ASSERT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}
