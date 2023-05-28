uint read_pixel(__global char* img, int2 loc, int2 size){
    if(all(loc >= 0) && all(loc < size)){
        return img[loc.x + loc.y*size.x];
    }
    return 0;
}

void write_pixel(__global char* img, char val, int2 loc, int2 size){
    if(all(loc>=0) && all(loc < size)){
        img[loc.x + loc.y*size.x] = val;
    }
}

/*
 * gray
 * 
 * get 2d image, return flattened image have one gray channel. 
 */
__kernel void gray(__read_only image2d_t src, __global char* dst, int width, int height){
    sampler_t _sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

    int2 loc = (int2)(get_global_id(0),get_global_id(1));

    float4 pixel = convert_float4(read_imageui(src,_sampler,loc)) / 255.0f;
    pixel.xyz = pixel.x * 0.72f + pixel.y* 0.21f + pixel.z*0.07f;
    pixel.z = 1.0f;
    pixel *= 255;

    dst[loc.x+loc.y*width] = (uint) pixel.x;
}

// normalize

// dynamicThreshold
__kernel void dynamicThreshold(__global char* src, __global char* dst, int width, int height, int blockSize){
    int2 loc = (int2)(get_global_id(0),get_global_id(1));
    int2 size = (int2)(width,height);

    int halfBlockSize = blockSize / 2;

    uint pixel = read_pixel(src,loc,size);

    float sum = 0;
    for(int x = loc.x - halfBlockSize; x <= loc.x + halfBlockSize;++x){
        for(int y = loc.y - halfBlockSize; y <= loc.y + halfBlockSize;++y){
            sum += read_pixel(src,(int2)(x,y),size);
        }
    }

    float mean = sum / ((2*halfBlockSize+1)*(2*halfBlockSize+1));
    pixel = pixel > mean ? 255 : 0;

    write_pixel(dst,pixel,loc,size);
}

// gaussian

// sovelX

// sovelY

// rosenfieldThinning4

// 