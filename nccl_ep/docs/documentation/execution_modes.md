# Execution Modes

Both `ncclEpDispatch()` and `ncclEpCombine()` operations support synchronous and staged semantics.

## Synchronous Mode (Default)

Corresponds to a synchronous version of an operation where GPU resources are allocated during the whole operation execution
including the time to wait for the data to be received. This mode doesn't allow for
computation/communication overlap.

```c
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
// Dispatch is complete when this returns
```

## Staged Mode (Low Latency Only)

In this mode, the operation is split into the send and receive phases enabling computation computation/communication overlap:
  * The operation is invoked with `send_only = true` flag
  * In this case, after all data transfers are initiated, the corresponding GPU resources are released
  * This enables an application to utilize all avaialble GPU resources for computation while offloaded data transfers are performed.
  * To complete the operation, the `ncclEpComplete()` primitive is used.

`send_only` is a collective property of the call: all ranks in the group must
pass the same value for a given dispatch or combine. Mixing staged and
synchronous calls across ranks for the same operation is unsupported and results
in undefined behavior, typically a timeout.

This mode is particularly beneficial for inference with multiple micro-batches.

```c
// Stage 1: Post send requests without waiting for completion
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
dispatch_cfg.send_only = 1;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
// Returns after initiating the operations

// Stage 2: Continue other computations...

// Stage 3: Wait for actual completion
ncclEpComplete(handle, /*config=*/NULL, stream);
// Now all data is actually sent/received
```
