#include <cstdint>
#include <cmath>
#include <emscripten.h>

extern "C" EMSCRIPTEN_KEEPALIVE
void color_map(int count, const uint8_t* rgb, int size, int channels,
               const float* lut, const float* linear, float* out) {
    const int strideB = channels, strideG = size*channels, strideR = size*strideG;
    for (int p=0; p<count; ++p) {
        const float r=linear[rgb[3*p]], g=linear[rgb[3*p+1]], b=linear[rgb[3*p+2]];
        const int ri=fminf(floorf(r),size-2), gi=fminf(floorf(g),size-2), bi=fminf(floorf(b),size-2);
        const float x=r-ri, y=g-gi, z=b-bi;
        const int i=ri*strideR+gi*strideG+bi*strideB;
        for (int c=0; c<channels; ++c) {
            const float a=lut[i+c]*(1-z)+lut[i+strideB+c]*z;
            const float d=lut[i+strideG+c]*(1-z)+lut[i+strideG+strideB+c]*z;
            const float e=lut[i+strideR+c]*(1-z)+lut[i+strideR+strideB+c]*z;
            const float f=lut[i+strideR+strideG+c]*(1-z)+lut[i+strideR+strideG+strideB+c]*z;
            out[p*channels+c]=(a*(1-y)+d*y)*(1-x)+(e*(1-y)+f*y)*x;
        }
    }
}
