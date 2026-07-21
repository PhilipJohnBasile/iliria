/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
#ifndef ILI_BACKEND_CUDA_H
#define ILI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ILI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct IliCudaTensor IliCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
int ili_cuda_init(const int *devices, int count);
void ili_cuda_shutdown(void);
int ili_cuda_device_count(void);
int ili_cuda_device_at(int index);
int ili_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
void ili_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);

/* Upload without executing, so capacity failures happen during model startup. */
int ili_cuda_tensor_upload(IliCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
int ili_cuda_matmul(IliCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

void ili_cuda_tensor_free(IliCudaTensor *tensor);
size_t ili_cuda_tensor_bytes(const IliCudaTensor *tensor);
int ili_cuda_tensor_device(const IliCudaTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
