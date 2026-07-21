// Metal 4 runtime probe for the macOS 27 / Xcode 27 path.
//
// This intentionally uses the same command primitives needed by the streamed
// MoE backend: legacy compute pipeline states, argument-table snapshots,
// dispatch-to-dispatch visibility, explicit residency, reusable command
// storage, and completion feedback with GPU error reporting.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <stdio.h>

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
#define ILI_METAL4_SDK 1
#else
#define ILI_METAL4_SDK 0
#endif

static const char *PROBE_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void probe_twice(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < count) output[i] = 2.0f * input[i];
}

kernel void probe_plus_one(device float *output [[buffer(0)]],
                           constant uint &count [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
  if (i < count) output[i] += 1.0f;
}
)METAL";

static int fail(const char *what, NSError *error) {
  fprintf(stderr, "metal4 probe: %s%s%s\n", what,
          error ? ": " : "",
          error ? [[error localizedDescription] UTF8String] : "");
  return 1;
}

int main(void) {
#if ILI_METAL4_SDK
  @autoreleasepool {
    if (@available(macOS 26.0, *)) {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) return fail("no Metal device", nil);

      NSError *error = nil;
      NSString *source = [NSString stringWithUTF8String:PROBE_SHADER];
      id<MTLLibrary> library = [device newLibraryWithSource:source
                                                    options:nil
                                                      error:&error];
      if (!library) return fail("shader compilation failed", error);

      id<MTLFunction> twiceFunction = [library newFunctionWithName:@"probe_twice"];
      id<MTLFunction> plusOneFunction = [library newFunctionWithName:@"probe_plus_one"];
      if (!twiceFunction || !plusOneFunction) return fail("shader function missing", nil);

      error = nil;
      id<MTLComputePipelineState> twicePipeline =
          [device newComputePipelineStateWithFunction:twiceFunction error:&error];
      if (!twicePipeline) return fail("twice pipeline creation failed", error);
      error = nil;
      id<MTLComputePipelineState> plusOnePipeline =
          [device newComputePipelineStateWithFunction:plusOneFunction error:&error];
      if (!plusOnePipeline) return fail("plus-one pipeline creation failed", error);

      id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
      id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
      id<MTL4CommandBuffer> commandBuffer = [device newCommandBuffer];
      if (!queue || !allocator || !commandBuffer)
        return fail("Metal 4 command objects unavailable", nil);

      MTL4ArgumentTableDescriptor *tableDescriptor = [MTL4ArgumentTableDescriptor new];
      tableDescriptor.maxBufferBindCount = 3;
      tableDescriptor.initializeBindings = YES;
      error = nil;
      id<MTL4ArgumentTable> table =
          [device newArgumentTableWithDescriptor:tableDescriptor error:&error];
      if (!table) return fail("argument table creation failed", error);

      static const uint32_t count = 257;
      const NSUInteger bytes = (NSUInteger)count * sizeof(float);
      id<MTLBuffer> input = [device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
      id<MTLBuffer> output = [device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
      id<MTLBuffer> constants = [device newBufferWithBytes:&count
                                                    length:sizeof(count)
                                                   options:MTLResourceStorageModeShared];
      if (!input || !output || !constants) return fail("probe buffer allocation failed", nil);

      MTLResidencySetDescriptor *residencyDescriptor = [MTLResidencySetDescriptor new];
      residencyDescriptor.initialCapacity = 3;
      error = nil;
      id<MTLResidencySet> residency =
          [device newResidencySetWithDescriptor:residencyDescriptor error:&error];
      if (!residency) return fail("residency set creation failed", error);
      id<MTLAllocation> allocations[] = {input, output, constants};
      [residency addAllocations:allocations count:3];
      [residency commit];

      id<MTLSharedEvent> completionEvent = [device newSharedEvent];
      MTL4CommitOptions *commitOptions = [MTL4CommitOptions new];
      dispatch_semaphore_t feedbackReady = dispatch_semaphore_create(0);
      if (!completionEvent || !commitOptions || !feedbackReady)
        return fail("completion objects unavailable", nil);

      __block NSError *feedbackError = nil;
      __block CFTimeInterval gpuStart = 0.0;
      __block CFTimeInterval gpuEnd = 0.0;
      MTL4CommitFeedbackHandler feedbackHandler = ^(id<MTL4CommitFeedback> feedback) {
        feedbackError = feedback.error;
        gpuStart = feedback.GPUStartTime;
        gpuEnd = feedback.GPUEndTime;
        dispatch_semaphore_signal(feedbackReady);
      };

      for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        // Reset is legal only after the prior cycle's event and feedback waits.
        if (cycle != 0) [allocator reset];

        float *inputValues = (float *)input.contents;
        float *outputValues = (float *)output.contents;
        const float cycleOffset = (float)cycle * 0.5f;
        for (uint32_t i = 0; i < count; ++i) {
          inputValues[i] = (float)i + cycleOffset;
          outputValues[i] = -1.0f;
        }

        feedbackError = nil;
        gpuStart = 0.0;
        gpuEnd = 0.0;

        // Reuse the same command buffer and allocator on the second cycle.
        [commandBuffer beginCommandBufferWithAllocator:allocator];
        // beginCommandBuffer clears prior per-buffer residency declarations.
        [commandBuffer useResidencySet:residency];
        id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return fail("compute encoder creation failed", nil);

        [encoder setArgumentTable:table];
        [table setAddress:input.gpuAddress atIndex:0];
        [table setAddress:output.gpuAddress atIndex:1];
        [table setAddress:constants.gpuAddress atIndex:2];
        [encoder setComputePipelineState:twicePipeline];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        // The first dispatch must retain its table snapshot while these bindings
        // are repurposed for the dependent, in-place second dispatch.
        [encoder barrierAfterEncoderStages:MTLStageDispatch
                       beforeEncoderStages:MTLStageDispatch
                         visibilityOptions:MTL4VisibilityOptionDevice];
        [table setAddress:output.gpuAddress atIndex:0];
        [table setAddress:constants.gpuAddress atIndex:1];
        [encoder setComputePipelineState:plusOnePipeline];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder endEncoding];
        [commandBuffer endCommandBuffer];

        // Commit feedback handlers are consumed by a commit. Re-add the same
        // stored block each cycle so errors and GPU timestamps remain observable.
        [commitOptions addFeedbackHandler:feedbackHandler];
        [queue commit:&commandBuffer count:1 options:commitOptions];
        const uint64_t completionValue = (uint64_t)cycle + 1;
        [queue signalEvent:completionEvent value:completionValue];

        const dispatch_time_t feedbackTimeout =
            dispatch_time(DISPATCH_TIME_NOW, 30ull * NSEC_PER_SEC);
        if (dispatch_semaphore_wait(feedbackReady, feedbackTimeout) != 0)
          return fail("timed out waiting for commit feedback", nil);
        if (feedbackError) return fail("GPU execution failed", feedbackError);
        if (gpuEnd < gpuStart) return fail("invalid GPU timestamps", nil);
        if (![completionEvent waitUntilSignaledValue:completionValue timeoutMS:30000])
          return fail("timed out waiting for the GPU event", nil);

        for (uint32_t i = 0; i < count; ++i) {
          const float expected = 2.0f * inputValues[i] + 1.0f;
          if (outputValues[i] != expected) {
            fprintf(stderr,
                    "metal4 probe: cycle %u mismatch at %u: got %.9g, expected %.9g\n",
                    cycle, i, outputValues[i], expected);
            return 1;
          }
        }
      }

      fprintf(stderr, "metal4 probe: two-cycle compute validation passed\n");
      return 0;
    }

    return fail("requires macOS 26 or newer", nil);
  }
#else
  fprintf(stderr, "metal4 probe: requires the macOS 26 SDK or newer\n");
  return 2;
#endif
}
