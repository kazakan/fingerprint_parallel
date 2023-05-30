uint read_pixel(__global uchar *img, int2 loc, int2 size) {
    if (all(loc >= 0) && all(loc < size)) {
        return img[loc.x + loc.y * size.x];
    }
    return 0;
}

void write_pixel(__global uchar *img, char val, int2 loc, int2 size) {
    if (all(loc >= 0) && all(loc < size)) {
        img[loc.x + loc.y * size.x] = val;
    }
}

/**
 * @brief get 2d image, return flattened image have one gray channel.
 *
 */
__kernel void gray(__read_only image2d_t src, __global uchar *dst, int width, int height) {
    sampler_t _sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    float4 pixel = convert_float4(read_imageui(src, _sampler, loc)) / 255.0f;
    pixel.xyz = pixel.x * 0.72f + pixel.y * 0.21f + pixel.z * 0.07f;
    pixel.z = 1.0f;
    pixel *= 255;

    write_pixel(dst, pixel.x, loc, size);
}

// normalize
__kernel void normalize(__global uchar *src, __global uchar *dst, float M, float V, float M0, float V0, int width, int height) {
    sampler_t _sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);
    float pixel = read_pixel(src, loc, size);

    float delta = sqrt(V0 * (pixel - M) * (pixel - M) / V);
    pixel = pixel > M ? M0 + delta : M0 - delta;

    uchar pixelByte = 0;
    if (pixel > 255)
        pixelByte = 255;
    else if (pixel < 0)
        pixelByte = 0;
    else
        pixelByte = pixel;

    write_pixel(dst, pixelByte, loc, size);
}

// dynamicThreshold
__kernel void dynamicThreshold(__global uchar *src, __global uchar *dst, int width, int height, int blockSize) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    int halfBlockSize = blockSize / 2;

    uint pixel = read_pixel(src, loc, size);

    float sum = 0;
    for (int x = loc.x - halfBlockSize; x <= loc.x + halfBlockSize; ++x) {
        for (int y = loc.y - halfBlockSize; y <= loc.y + halfBlockSize; ++y) {
            sum += read_pixel(src, (int2)(x, y), size);
        }
    }

    float mean = sum / ((2 * halfBlockSize + 1) * (2 * halfBlockSize + 1));
    pixel = pixel > mean ? 255 : 0;

    write_pixel(dst, pixel, loc, size);
}

// gaussian
__kernel void gaussian(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    // 121
    // 242
    // 121

    uint val = read_pixel(src, loc + (int2)(-1, -1), size) * 1 +
               read_pixel(src, loc + (int2)(-1, 0), size) * 2 +
               read_pixel(src, loc + (int2)(-1, +1), size) * 1 +
               read_pixel(src, loc + (int2)(0, -1), size) * 2 +
               read_pixel(src, loc + (int2)(0, 0), size) * 4 +
               read_pixel(src, loc + (int2)(0, +1), size) * 2 +
               read_pixel(src, loc + (int2)(+1, -1), size) * 1 +
               read_pixel(src, loc + (int2)(+1, 0), size) * 2 +
               read_pixel(src, loc + (int2)(+1, +1), size) * 1;

    // rount(a/b) = (a + (b/2)) / b
    val = (val + 8) / 16;

    write_pixel(dst, val, loc, size);
}

// sobelX
__kernel void sobelX(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    // 1 0 -1
    // 2 0 -2
    // 1 0 -1

    int val = read_pixel(src, loc + (int2)(-1, -1), size) * 1 +
              read_pixel(src, loc + (int2)(-1, 0), size) * 2 +
              read_pixel(src, loc + (int2)(-1, +1), size) * 1 +
              read_pixel(src, loc + (int2)(+1, -1), size) * -1 +
              read_pixel(src, loc + (int2)(+1, 0), size) * -2 +
              read_pixel(src, loc + (int2)(+1, +1), size) * -1;

    // set value in range 0~255
    val = val > 255 ? 255 : val;
    val = val < 0 ? 0 : val;

    write_pixel(dst, val, loc, size);
}

// sobelY
__kernel void sobelY(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    //  1  2  1
    //  0  0  0
    // -1 -2 -1

    uint val = read_pixel(src, loc + (int2)(-1, -1), size) * 1 +
               read_pixel(src, loc + (int2)(-1, +1), size) * -1 +
               read_pixel(src, loc + (int2)(0, -1), size) * 2 +
               read_pixel(src, loc + (int2)(0, +1), size) * -2 +
               read_pixel(src, loc + (int2)(+1, -1), size) * 1 +
               read_pixel(src, loc + (int2)(+1, +1), size) * 1;

    // set value in range 0~255
    val = val > 255 ? 255 : val;
    val = val < 0 ? 0 : val;

    write_pixel(dst, val, loc, size);
}

// Rosenfield Thinning Four connectivity One iteration
__kernel void rosenfieldThinFourCon(
    __global uchar *src,
    __global uchar *dst,
    int width,
    int height,
    int dir, // N,E,S,W = 0,1,2,3
    __global uchar *globalContinueFlags,
    __local uchar *localContinueFlags) {

    const int2 loc = (int2)(get_global_id(0), get_global_id(1));
    const int2 size = (int2)(width, height);
    const int2 localLoc = (int2)(get_local_id(0), get_local_id(1));
    const int2 groupSize = (int2)(get_local_size(0), get_local_size(1));
    const int2 groupId = (int2)(get_group_id(0), get_group_id(1));
    const int2 numGroups = (int2)(get_num_groups(0), get_num_groups(1));
    const int N = groupSize.x * groupSize.y;
    const int localIdx = localLoc.x + localLoc.y * groupSize.x;

    uchar pixel = read_pixel(src, loc, size);

    bool changed = false;

    if (pixel > 0) {
        // neighbors (N,NE,E,SE,S,SW,W,NW)
        uchar neighbors = 0;
        neighbors |= (((read_pixel(src, loc + (int2)(0, -1), size) ? 1 : 0) << 7));
        neighbors |= (((read_pixel(src, loc + (int2)(1, -1), size) ? 1 : 0) << 6));
        neighbors |= (((read_pixel(src, loc + (int2)(1, 0), size) ? 1 : 0) << 5));
        neighbors |= (((read_pixel(src, loc + (int2)(1, 1), size) ? 1 : 0) << 4));
        neighbors |= (((read_pixel(src, loc + (int2)(0, 1), size) ? 1 : 0) << 3));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, 1), size) ? 1 : 0) << 2));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, 0), size) ? 1 : 0) << 1));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, -1), size) ? 1 : 0) << 0));

        // number of 4 connected neighbors
        uchar n4Neighbors = (neighbors & 0x80 ? 1 : 0) + (neighbors & 0x20 ? 1 : 0) + (neighbors & 0x08 ? 1 : 0) + (neighbors & 0x02 ? 1 : 0);

        switch (dir) {
        case 0: // N
            if (n4Neighbors == 2) {
                changed = (neighbors == 0b00111000) || (neighbors == 0b00001110);
            } else if (n4Neighbors == 3) {
                changed = (neighbors == 0b00111110);
            }
            break;

        case 1: // E
            if (n4Neighbors == 2) {
                changed = (neighbors == 0b10000011) || (neighbors == 0b00001110);
            } else if (n4Neighbors == 3) {
                changed = (neighbors == 0b10001111);
            }
            break;

        case 2: // S
            if (n4Neighbors == 2) {
                changed = (neighbors == 0b10000011);
            } else if (n4Neighbors == 3) {
                changed = (neighbors == 0b11100011);
            }
            break;

        case 3: // W
            if (n4Neighbors == 2) {
                changed = (neighbors == 0b11100000) || (neighbors == 0b00111000);
            } else if (n4Neighbors == 3) {
                changed = (neighbors == 0b11111000);
            }
            break;
        }

        // if meet condition then change, else don't change
        pixel = changed ? 0 : pixel;
    }

    // write to dst
    write_pixel(dst, pixel, loc, size);

    // check at least one pixel changed
    localContinueFlags[localIdx] = changed;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = N >> 1; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            localContinueFlags[localIdx] |= localContinueFlags[localIdx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write whether pixel changed in work group
    if (localIdx == 0) {
        globalContinueFlags[groupId.x + groupId.y * numGroups.x] = localContinueFlags[0];
    }
}

// Rosenfield Thinning Eight connectivity One iteration
__kernel void rosenfieldThinEightCon(
    __global uchar *src,
    __global uchar *dst,
    int width,
    int height,
    int dir,
    __global uchar *globalContinueFlags,
    __local uchar *localContinueFlags) {

    const int2 loc = (int2)(get_global_id(0), get_global_id(1));
    const int2 size = (int2)(width, height);
    const int2 localLoc = (int2)(get_local_id(0), get_local_id(1));
    const int2 groupSize = (int2)(get_local_size(0), get_local_size(1));
    const int2 groupId = (int2)(get_group_id(0), get_group_id(1));
    const int2 numGroups = (int2)(get_num_groups(0), get_num_groups(1));
    const int N = groupSize.x * groupSize.y;
    const int localIdx = localLoc.x + localLoc.y * groupSize.x;

    uchar pixel = read_pixel(src, loc, size);

    bool changed = false;

    if (pixel > 0) {
        // neighbors (N,NE,E,SE,S,SW,W,NW)
        uchar neighbors = 0;
        neighbors |= (((read_pixel(src, loc + (int2)(0, -1), size) ? 1 : 0) << 7));
        neighbors |= (((read_pixel(src, loc + (int2)(1, -1), size) ? 1 : 0) << 6));
        neighbors |= (((read_pixel(src, loc + (int2)(1, 0), size) ? 1 : 0) << 5));
        neighbors |= (((read_pixel(src, loc + (int2)(1, 1), size) ? 1 : 0) << 4));
        neighbors |= (((read_pixel(src, loc + (int2)(0, 1), size) ? 1 : 0) << 3));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, 1), size) ? 1 : 0) << 2));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, 0), size) ? 1 : 0) << 1));
        neighbors |= (((read_pixel(src, loc + (int2)(-1, -1), size) ? 1 : 0) << 0));

        // number of 8 connected neighbors
        uchar neighborsBits = neighbors;
        int n8Neighbors = 0;
        for (n8Neighbors = 0; neighborsBits; n8Neighbors++)
            neighborsBits &= neighborsBits - 1;

        uchar borderFlag = 0;
        switch (dir) {
        case 0: // N
            borderFlag = 0b10000000;
            break;
        case 1: // E
            borderFlag = 0b00100000;
            break;
        case 2: // S
            borderFlag = 0b00001000;
            break;
        case 3: // W
            borderFlag = 0b00000010;
            break;
        }

        if ((neighbors & borderFlag) != 0) {
            if ((n8Neighbors > 1) && (n8Neighbors < 7)) {
                uchar pattern = (1 << n8Neighbors) - 1;
                changed = true;
                for (int i = 0; i < 8 - n8Neighbors; ++i) {
                    if (neighbors == pattern) {
                        changed = false;
                        break;
                    };
                    pattern <<= 1;
                }
            } else if (n8Neighbors == 8 || n8Neighbors == 7) {
                changed = true;
            }
        }

        // if meet condition then change, else don't change
        pixel = changed ? 0 : pixel;
    }

    // write to dst
    write_pixel(dst, pixel, loc, size);

    // check at least one pixel changed
    localContinueFlags[localIdx] = changed;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = N >> 1; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            localContinueFlags[localIdx] |= localContinueFlags[localIdx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write whether pixel changed in work group
    if (localIdx == 0) {
        globalContinueFlags[groupId.x + groupId.y * numGroups.x] = localContinueFlags[0];
    }
}

// crossNumbers
__kernel void crossNumbers(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    // neighbors (N,NE,E,SE,S,SW,W,NW)
    uchar neighbors = 0;
    neighbors |= (((read_pixel(src, loc + (int2)(0, -1), size) ? 1 : 0) << 7));
    neighbors |= (((read_pixel(src, loc + (int2)(1, -1), size) ? 1 : 0) << 6));
    neighbors |= (((read_pixel(src, loc + (int2)(1, 0), size) ? 1 : 0) << 5));
    neighbors |= (((read_pixel(src, loc + (int2)(1, 1), size) ? 1 : 0) << 4));
    neighbors |= (((read_pixel(src, loc + (int2)(0, 1), size) ? 1 : 0) << 3));
    neighbors |= (((read_pixel(src, loc + (int2)(-1, 1), size) ? 1 : 0) << 2));
    neighbors |= (((read_pixel(src, loc + (int2)(-1, 0), size) ? 1 : 0) << 1));
    neighbors |= (((read_pixel(src, loc + (int2)(-1, -1), size) ? 1 : 0) << 0));

    uchar rotated = (neighbors >> 1) | ((neighbors & 1) << 7);
    uchar crossed = (rotated ^ neighbors);

    char count = 0;
    for (count = 0; crossed; count++)
        crossed &= crossed - 1;

    count >>= 1;

    write_pixel(dst, count, loc, size);
}