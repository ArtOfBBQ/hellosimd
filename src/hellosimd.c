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

#define FASTEST_SOLUTION_ONLY

/*
We need to pass multiple arguments to the entry point when we
start a thread, but the arguments are expected to be of type
"void *".

By storing our arguments in this struct we can pass them. It's
convoluted but no different from passing function arguments
in single-threaded cocde.
*/
typedef struct handle_chunk_args {
    float * vector_1;
    float * vector_2;
    float * results;
    uint64_t starting_i;
    uint64_t last_i;
} handle_chunk_args;

#ifndef FASTEST_SOLUTION_ONLY
/* Use this to have a thread handle a chunk without simd */
static void * handle_chunk(void * arguments)
{
    // cast the void * back to a "handle_chunk_args" struct
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
#endif

/* Use this to have a thread handle a chunk of data with simd */
static void * handle_chunk_with_simd(void * arguments)
{
    __m256 v1;
    __m256 v2;
    
    // cast the void * back to a "handle_chunk_args" struct
    handle_chunk_args * args =
        (handle_chunk_args *)arguments;
    
    for (
        uint64_t i = args->starting_i;
        i < (args->last_i - 9);
        i += 8)
    {
        v1 = _mm256_load_ps( (args->vector_1 + i) );
        v2 = _mm256_load_ps( (args->vector_2 + i) );
        v1 = _mm256_add_ps(v1, v2);
        _mm256_store_ps(
            args->results + i,
            _mm256_mul_ps(v1, v2));
    }
    
    return NULL;
}

/* get time elapsed between start & end */
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
   
    #ifndef FASTEST_SOLUTION_ONLY 
    long vanilla_time_used;
    long simd_time_used;
    long vanilla_threaded_time_used;
    #endif
    long simd_threaded_time_used;
    
    printf("Hello, simd instructions!\n");
    /*
    Step 1: prepare 2 big vectors with whatever values
    to add, then multiply.
    */
    srand((uint32_t)time(NULL));
    uint64_t vectors_size = 50000000; //12500000 * ((rand() % 10) + 5);
    uint32_t threads_size = 4;
    uint64_t chunk_size =
        (vectors_size / threads_size) + 1;
    chunk_size -= (chunk_size % 8);
    printf(
        "Will add, then mull 2 vecs of %llu floats each...\n",
        vectors_size);
    printf("allocate memory...\n");
    float * vector_1 =
        malloc(sizeof(float) * vectors_size);
    float * vector_2 =
        malloc(sizeof(float) * vectors_size);
    #ifndef FASTEST_SOLUTION_ONLY
    float * results =
        malloc(sizeof(float) * vectors_size);
    float * simd_results =
        malloc(sizeof(float) * vectors_size);
    float * vanilla_threaded_results =
        malloc(sizeof(float) * vectors_size);
    #endif
    float * simd_threaded_results =
        malloc(sizeof(float) * vectors_size);
    
    printf("fill in input vectors with random values...\n");
    for (uint64_t i = 0; i < vectors_size; i++) {
        vector_1[i] = rand();
        vector_2[i] = i % 4 * 0.54f;
    }
    
    #ifndef FASTEST_SOLUTION_ONLY
    // Step 2: calculate vector addition and multiplication
    // in the simplest possible way
    printf("Computing with 1 thread - no simd...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (uint64_t i = 0; i < vectors_size; i++) {
        results[i] = (vector_1[i] + vector_2[i]) * vector_2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    vanilla_time_used = to_microsecs(time_diff(start, end)); 
    
    // Step 3: calculate the exact same vector addition with simd
    printf("Computing with 1 thread using simd...\n");
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
    // cpu threads (no simd)
    printf(
        "Computing with %u threads - no simd...\n",
        threads_size);
    pthread_t * threads =
        malloc(sizeof(pthread_t) * threads_size);
    handle_chunk_args * thread_args =
        malloc(sizeof(handle_chunk_args) * 4);
    
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
        
        int result = pthread_create(
            /* pthread_t * thread    : */ &(threads[i]),
            /* pthread_attr_t * attr : */ NULL,
            /* void *(*start_routine)(void *) : */ &handle_chunk,
            /* void * arg : */ this_thread_args);
        if (result) {
            printf("ERROR during creation of thread: %u\n", i);
        }
    }
    
    for (
        uint32_t i = 0;
        i < threads_size;
        i += 1)
    {
        int result = pthread_join(threads[i], NULL);
        if (result) {
            printf(
                "ERROR while joining (non-simd) thread: %u\n",
                i);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    vanilla_threaded_time_used =
        to_microsecs(time_diff(start, end));
    #endif
    
    // step 5: calculate the same vector addition with multiple
    // cpu threads, with simd on each thread
    printf(
        "Computing with %u threads and using simd...\n",
        threads_size);
    pthread_t * simd_threads =
        malloc(sizeof(pthread_t) * threads_size);
    handle_chunk_args * simd_thread_args =
        malloc(sizeof(handle_chunk_args) * 4);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (
        uint32_t i = 0;
        i < threads_size;
        i += 1)
    {
        simd_thread_args[i].vector_1 = vector_1;
        simd_thread_args[i].vector_2 = vector_2;
        simd_thread_args[i].results = simd_threaded_results;
        simd_thread_args[i].starting_i = i * chunk_size;
        simd_thread_args[i].last_i =
            (i + 1) * chunk_size > vectors_size ?
                    vectors_size : (i + 1) * chunk_size;
        
        void * this_thread_args = (void *)&simd_thread_args[i];
        
        int result = pthread_create(
            /* pthread_t * thread             : */
                &(simd_threads[i]),
            /* pthread_attr_t * attr          : */
                NULL,
            /* void *(*start_routine)(void *) : */
                &handle_chunk_with_simd,
            /* void * arg : */
                this_thread_args);
        if (result) {
            printf("ERROR while creating simd thread: %u\n", i);
        }
    }
    
    for (
        uint32_t i = 0;
        i < threads_size;
        i += 1)
    {
        int result = pthread_join(simd_threads[i], NULL);
        if (result) {
            printf("ERROR while joining simd thread: %u\n", i);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    simd_threaded_time_used =
        to_microsecs(time_diff(start, end));
    
    // report results 
    #ifndef FASTEST_SOLUTION_ONLY
    printf(
        "%ld microseconds taken by 'naive' code\n",
        vanilla_time_used);
    printf(
        "%ld microseconds taken using simd\n",
        simd_time_used);
    printf(
        "%ld microseconds taken with 4 cpu threads\n",
        vanilla_threaded_time_used);
    #endif
    printf(
        "%ld microseconds taken by 4 simd threads\n",
        simd_threaded_time_used);
   
    #ifndef FASTEST_SOLUTION_ONLY 
    // step 5: spot check all calculation results equal
    for (
        uint64_t i = 0;
        i < (vectors_size - 200);
        i += 200)
    {
         assert(
             (results[i] - simd_results[i]) < 0.05f);
         assert(
             (results[i] - simd_results[i]) > -0.05f);
         assert(
             (results[i] - vanilla_threaded_results[i]) < 0.05f);
         assert(
             (results[i] - vanilla_threaded_results[i]) > -0.05f);
         assert(
             (results[i] - simd_threaded_results[i]) > -0.05f);
         assert(
             (results[i] - simd_threaded_results[i]) > -0.05f);
    }

    free(results);
    free(simd_results);
    free(vanilla_threaded_results);
    #endif
    free(simd_threaded_results);
    
    return 0;
}

