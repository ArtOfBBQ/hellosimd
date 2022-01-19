/*
Let's get our feet wet with using simd.
*/

#include "inttypes.h"
#include "stdlib.h"
#include "immintrin.h" // intel 'intrinsics' nescessary for simd
#include "time.h"      // time which is faster
#include "stdio.h"
#include "assert.h"

int main(int argc, const char * argv[])
{
    clock_t start, end;
    double vanilla_time_used;
    double simd_time_used;
    
    printf("Hello, simd instructions!\n");
    
    /*
    Step 1: prepare 2 big vectors with whatever values
    to add to eachother.
    */
    uint64_t vectors_size = 210000000;
    printf(
        "Will add 2 vectors of %llu floats each...\n",
        vectors_size);
    float * vector_1 =
        malloc(sizeof(float) * vectors_size);
    float * vector_2 =
        malloc(sizeof(float) * vectors_size);
    float * results =
        malloc(sizeof(float) * vectors_size);
    assert(sizeof(float) == 4);

    for (uint64_t i = 0; i < vectors_size; i++) {
        vector_1[i] = i % 15 * 0.23f;
        vector_2[i] = i % 4 * 0.54f;
    }
    
    // Step 2: calculate vector addition in the simplest possible
    // "naive" way
    start = clock();
    for (uint64_t i = 0; i < vectors_size; i++) {
        results[i] = vector_1[i] + vector_2[i];
    }
    end = clock();
    vanilla_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Step 3: calculate the same vector addition with simd
    start = clock();
    __m128 v1;
    __m128 v2;
    __m128 sums;
    float * simd_results =
        malloc(sizeof(float) * vectors_size);
    for (uint64_t i = 0; i < vectors_size; i += 4) {
        v1 = _mm_load_ps( (vector_1 + i) );
        v2 = _mm_load_ps( (vector_2 + i) );
        sums = _mm_add_ps(v1, v2);
        _mm_store_ps(
            simd_results + i,
            sums);
    }
    end = clock();
    simd_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("'naive' code time used: %f\n", vanilla_time_used);
    printf("simd code time used: %f\n", simd_time_used);
    
    // step 4: prove all calculation results are equal
    for (uint64_t i = 0; i < vectors_size; i++) {
        assert((results[i] - simd_results[i]) < 0.05f);
        assert((results[i] - simd_results[i]) > -0.05f);
    }
    
    return 0;
}

