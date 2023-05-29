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

/*
 * gray
 *
 * get 2d image, return flattened image have one gray channel.
 */
__kernel void gray(__read_only image2d_t src, __global uchar *dst, int width, int height) {
    sampler_t _sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    float4 pixel = convert_float4(read_imageui(src, _sampler, loc)) / 255.0f;
    pixel.xyz = pixel.x * 0.72f + pixel.y * 0.21f + pixel.z * 0.07f;
    pixel.z = 1.0f;
    pixel *= 255;

    write_pixel(dst,pixel.x,loc,size);
}

// normalize

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
__kernel void rosenfieldThinFourCon(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    // neighbors (N,NE,E,SE,S,SW,W,NW)
    char8 neighbors = (char8)(read_pixel(src, loc + (int2)(0, -1), size),
                              read_pixel(src, loc + (int2)(1, -1), size),
                              read_pixel(src, loc + (int2)(1, 0), size),
                              read_pixel(src, loc + (int2)(1, 1), size),
                              read_pixel(src, loc + (int2)(0, 1), size),
                              read_pixel(src, loc + (int2)(-1, 1), size),
                              read_pixel(src, loc + (int2)(-1, 0), size),
                              read_pixel(src, loc + (int2)(-1, -1), size));

    // write_pixel(dst, val, loc, size);
    // TODO : Implement
}

// crossNumbers
__kernel void crossNumbers(__global uchar *src, __global uchar *dst, int width, int height) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(width, height);

    // neighbors (N,NE,E,SE,S,SW,W,NW)
    char8 neighbors = (char8)(read_pixel(src, loc + (int2)(0, -1), size),
                              read_pixel(src, loc + (int2)(1, -1), size),
                              read_pixel(src, loc + (int2)(1, 0), size),
                              read_pixel(src, loc + (int2)(1, 1), size),
                              read_pixel(src, loc + (int2)(0, 1), size),
                              read_pixel(src, loc + (int2)(-1, 1), size),
                              read_pixel(src, loc + (int2)(-1, 0), size),
                              read_pixel(src, loc + (int2)(-1, -1), size));

    // write_pixel(dst, val, loc, size);
    // TODO : Implement
}