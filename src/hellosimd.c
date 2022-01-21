/*
Let's get our feet wet with using simd.
*/

#include "inttypes.h"
#include "stdlib.h"
#include "immintrin.h" // intel 'intrinsics' nescessary for simd
#include "time.h"      // time which is faster
#include "stdio.h"
#include "assert.h"
#include "pthread.h"

typedef struct handle_chunk_args {
    float * vector_1;
    float * vector_2;
    float * results;
    uint64_t starting_i;
    uint64_t last_i;
} handle_chunk_args;

static void * handle_chunk(void * arguments)
{
    handle_chunk_args * args =
        (handle_chunk_args *)arguments;

    for (
        uint64_t i = args->starting_i;
        i < args->last_i;
        i++)
    {
        args->results[i] =
            (args->vector_1[i] + args->vector_2[i])
                * (args->vector_2[i]);
    }
    
    return NULL;
}

static struct timespec time_diff(
    struct timespec start,
    struct timespec end)
{
    struct timespec temp;
    
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    
    return temp;
}

static long to_microsecs(struct timespec input) {
    return (input.tv_sec * 1000000) + (input.tv_nsec / 1000);
}

int main()
{
    struct timespec start, end;
    
    long vanilla_time_used;
    long simd_time_used;
    long vanilla_threaded_time_used;
    
    printf("Hello, simd instructions!\n");
    
    /*
    Step 1: prepare 2 big vectors with whatever values
    to add, then multiply.
    */
    srand(0);
    uint64_t vectors_size = 12500000 * ((rand() % 10) + 1);
    printf(
        "Will add, then multiply 2 vectors of %llu floats each...\n",
        vectors_size);
    float * vector_1 =
        malloc(sizeof(float) * vectors_size);
    float * vector_2 =
        malloc(sizeof(float) * vectors_size);
    float * results =
        malloc(sizeof(float) * vectors_size);
    float * simd_results =
        malloc(sizeof(float) * vectors_size);
    float * vanilla_threaded_results =
        malloc(sizeof(float) * vectors_size);
    
    for (uint64_t i = 0; i < vectors_size; i++) {
        vector_1[i] = rand();
        vector_2[i] = i % 4 * 0.54f;
    }
    
    // Step 2: calculate vector addition and multiplication
    // in the simplest possible way
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (uint64_t i = 0; i < vectors_size; i++) {
        results[i] = (vector_1[i] + vector_2[i]) * vector_2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    vanilla_time_used = to_microsecs(time_diff(start, end)); 
    
    // Step 3: calculate the exact same vector addition with simd
    __m256 v1;
    __m256 v2;
    clock_gettime(
        CLOCK_MONOTONIC,
        &start);
    for (uint64_t i = 0; i < vectors_size; i += 8) {
        v1 = _mm256_load_ps( (vector_1 + i) );
        v2 = _mm256_load_ps( (vector_2 + i) );
        v1 = _mm256_add_ps(v1, v2);
        _mm256_store_ps(
            simd_results + i,
            _mm256_mul_ps(v1, v2));
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    simd_time_used = to_microsecs(time_diff(start, end)); 
    
    // step 4: calculate the same vector addition with multiple
    // cpu threads
    uint32_t threads_size = 4;
    pthread_t * threads =
        malloc(sizeof(pthread_t) * threads_size);
    handle_chunk_args * thread_args =
        malloc(sizeof(handle_chunk_args) * 4);
    uint64_t chunk_size =
        (vectors_size / threads_size) + 1;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (
        uint32_t i = 0;
        i < threads_size;
        i += 1)
    {
        thread_args[i].vector_1 = vector_1;
        thread_args[i].vector_2 = vector_2;
        thread_args[i].results = vanilla_threaded_results;
        thread_args[i].starting_i = i * chunk_size;
        thread_args[i].last_i =
            (i + 1) * chunk_size > vectors_size ?
                    vectors_size : (i + 1) * chunk_size;
        
        void * this_thread_args = (void *)&thread_args[i];
        
        pthread_create(
            /* pthread_t * thread    : */ &(threads[i]),
            /* pthread_attr_t * attr : */ NULL,
            /* void *(*start_routine)(void *) : */ &handle_chunk,
            /* void * arg : */ this_thread_args);
    }
    
    for (
        uint32_t i = 0;
        i < threads_size;
        i += 1)
    {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    vanilla_threaded_time_used =
        to_microsecs(time_diff(start, end));
    
    printf(
        "'naive' code - microseconds: %ld\n",
        vanilla_time_used);
    printf(
        "using simd - microseconds: %ld\n",
        simd_time_used);
    printf(
        "4 cpu threads - microseconds: %ld\n",
        vanilla_threaded_time_used);
    
    // step 5: prove all calculation results are equal
    for (
        uint64_t i = 0;
        i < vectors_size;
        i++)
    {
         assert(
             (results[i] - simd_results[i]) < 0.05f);
         assert(
             (results[i] - simd_results[i]) > -0.05f);
         assert(
             (results[i] - vanilla_threaded_results[i]) < 0.05f);
         assert(
             (results[i] - vanilla_threaded_results[i]) > -0.05f);
    }
    
    return 0;
}

