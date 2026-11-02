# TP-Only `ft_nccl` Integration

Goal: use `ft_collective` / `ft_nccl` for vLLM tensor-parallel (TP)
all-reduce only. EP dispatch/combine is out of scope for now.

## Current TP Path

TP all-reduce enters vLLM here:

- `GroupCoordinator.all_reduce` in `vllm/distributed/parallel_state.py`
- `CudaCommunicator.all_reduce` in
  `vllm/distributed/device_communicators/cuda_communicator.py`

`CudaCommunicator.all_reduce` normally tries several CUDA-side paths before
falling back to `torch.distributed.all_reduce`:

1. NCCL symmetric-memory custom op
2. quick reduce
3. flashinfer all-reduce
4. custom all-reduce
5. torch symmetric-memory communicator
6. PyNCCL
7. torch distributed fallback

`ft_nccl` only participates in step 7, so the earlier paths must be bypassed
for TP if the goal is to route TP collectives through `ft_nccl`.

## Minimal Changes

1. Import `ft_collective` before distributed initialization.

   Importing `ft_collective` registers the `"ft_nccl"` torch distributed backend.
   This must happen before vLLM creates process groups.

2. Add a small enable flag.

   Example env flag:

   ```text
   VLLM_USE_FT_NCCL_TP=1
   ```

3. Create the TP device group with backend `"ft_nccl"`.

   The TP group is created in `initialize_model_parallel` when calling
   `init_model_parallel_group(..., group_name="tp")`. Under the new flag, pass
   `"ft_nccl"` for this group only. Leave world, DP, EP, PP, etc. on the normal
   backend.

4. Force TP all-reduce to reach the torch distributed fallback.

   For the first cut, disable the competing TP all-reduce paths:

   ```text
   VLLM_DISABLE_PYNCCL=1
   VLLM_ALLREDUCE_USE_SYMM_MEM=0
   VLLM_ALLREDUCE_USE_FLASHINFER=0
   --disable-custom-all-reduce
   ```

5. Use an FT symmetric tensor in the fallback path.

   This is the key correctness caveat. `FTProcessGroup` only uses the FT kernel
   when the input tensor was allocated with `FTProcessGroup.empty()`. A normal
   `input_.clone()` will fall back to regular NCCL inside `ft_nccl`.

   The TP fallback path needs to become conceptually:

   ```python
   ft_pg = get_ft_process_group_for(self.device_group)
   out = ft_pg.empty(*input_.shape, dtype=input_.dtype).reshape_as(input_)
   out.copy_(input_)
   torch.distributed.all_reduce(out, group=self.device_group)
   return out
   ```

   The existing `ft_collective.get_ft_process_group()` registry is rank-keyed,
   which is probably not precise enough once vLLM creates multiple groups. Add a
   way to retrieve the Python `FTProcessGroup` for the specific TP
   `device_group`, or otherwise cache it during TP group creation.

## Suggested First Milestone

Implement only this path:

- `VLLM_USE_FT_NCCL_TP=1`
- import/register `ft_collective`
- TP group backend becomes `"ft_nccl"`
- `CudaCommunicator.all_reduce` special-cases `unique_name.split(":")[0] == "tp"`
- TP all-reduce stages into `FTProcessGroup.empty()`
- call `torch.distributed.all_reduce` on the staged tensor

Everything else should stay on the existing vLLM communication paths.

## Out of Scope

- EP `all_gatherv`
- EP `reduce_scatterv`
- native FT reduce-scatter
- variadic FT all-gather
- continuing execution with partial-rank result masks

For the first TP-only version, treat an FT timeout as a fail-fast condition
unless a higher-level recovery policy is added later.
