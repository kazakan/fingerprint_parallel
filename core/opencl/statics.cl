// v array, s scalar, sp scalar pointer

#define REDUCTION(input_type, v_input, output_type, v_tmp, s_ret, s_inputSize, \
                  associative_assignment_ops, preprocess_ops)                  \
    do {                                                                       \
        const int global_id = get_global_id(0);                                \
        const int local_id = get_local_id(0);                                  \
        const int local_size = get_local_size(0);                              \
                                                                               \
        output_type partialResult = 0;                                         \
                                                                               \
        for (int i = local_id; i < inputSize; i += local_size) {               \
            associative_assignment_ops(partialResult,                          \
                                       preprocess_ops(v_input[i]));            \
        }                                                                      \
                                                                               \
        v_tmp[local_id] = partialResult;                                       \
                                                                               \
        for (unsigned int stride = local_size >> 1; stride > 0;                \
             stride >>= 1) {                                                   \
            barrier(CLK_LOCAL_MEM_FENCE);                                      \
            if (local_id < stride) {                                           \
                associative_assignment_ops(v_tmp[local_id],                    \
                                           v_tmp[local_id + stride]);          \
            }                                                                  \
        }                                                                      \
                                                                               \
        barrier(CLK_LOCAL_MEM_FENCE);                                          \
                                                                               \
        if (global_id == 0) {                                                  \
            s_ret = v_tmp[0];                                                  \
        }                                                                      \
    } while (0);

#define SUM_OPS(lhs, rhs) ((lhs) += (rhs))
#define OR_OPS(lhs, rhs) ((lhs) |= (rhs))

#define PRE_IDENTICAL(v) (v)
#define PRE_SQUARE(v) (((long)v) * (v))

#define OR_REDUCTION(input_type, v_input, output_type, v_tmp, s_ret, \
                     inputSize)                                      \
    REDUCTION(input_type, v_input, output_type, v_tmp, s_ret, inputSize, OR_OPS)

/**
 * @brief Calculate sum of elements in a work group.
 */
__kernel void sum(__global uchar *v_input, __global long *sp_output,
                  __local long *v_tmp, int inputSize) {
    REDUCTION(uchar, v_input, long, v_tmp, sp_output[0], inputSize, SUM_OPS,
              PRE_IDENTICAL);
}

/**
 * @brief Calculate sum of elements^2 in a work group.
 */
__kernel void squareSum(__global uchar *v_input, __global long *sp_output,
                        __local long *v_tmp, int inputSize) {
    REDUCTION(uchar, v_input, long, v_tmp, sp_output[0], inputSize, SUM_OPS,
              PRE_SQUARE);
}

__kernel void mean(__global uchar *v_input, __global float *sp_output,
                   __local long *v_tmp, int inputSize) {
    long sum = 0;
    REDUCTION(uchar, v_input, long, v_tmp, sum, inputSize, SUM_OPS,
              PRE_IDENTICAL);

    if (get_global_id(0) == 0) {
        sp_output[0] = ((float)sum) / inputSize;
    }
}

__kernel void var(__global uchar *v_input, __global float *sp_output,
                  __local long *v_tmp1, __local long *v_tmp2, int inputSize) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    long sum = 0;
    long squareSum = 0;

    long v = 0;
    for (int i = local_id; i < inputSize; i += local_size) {
        v = v_input[i];
        sum += v;
        squareSum += v * v;
    }

    v_tmp1[local_id] = sum;
    v_tmp2[local_id] = squareSum;

    for (unsigned int stride = local_size >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < stride) {
            v_tmp1[local_id] += v_tmp1[local_id + stride];
            v_tmp2[local_id] += v_tmp2[local_id + stride];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id == 0) {
        float mean = ((float)v_tmp1[0]) / inputSize;

        sp_output[0] = ((float)v_tmp2[0]) / inputSize - mean * mean;
    }
}
