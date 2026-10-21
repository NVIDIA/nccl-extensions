---
name: cpp-style
description: "Code style rules for C++/CUDA in nccl-extensions: no default parameter values, no nested ternaries, short comments, braced single-statement blocks, indented nested blocks, no inline asm outside device_primitives.cuh, reuse existing helpers, no duplication, readable function sizes, and fixes as new commits. Use whenever writing, editing or reviewing C++/CUDA code in this repo (nccl_ep, nccl_m2n)."
---

# C++/CUDA style

Apply these when writing or editing C++/CUDA, and flag violations when reviewing.
They sit on top of `.clang-format`, which handles whitespace and wrapping.
Apply the rules to new code and to lines you touch. Do not reformat unrelated code.

## 1. No default parameter values

Every parameter is passed explicitly at every call site.

```cpp
// WRONG
void launch_permute(int top_k, bool allow_overflow_drop = false);

// RIGHT
void launch_permute(int top_k, bool allow_overflow_drop);
```

- Applies to function and method signatures, including `__device__` and `__global__`.
- Template parameter defaults (`template <bool kFlag = false>`) are also avoided in new code.
- When adding a parameter, update every existing call site. Do not leave callers relying on a default.
- Default member initializers on data-only structs (`= nullptr`, `= false` in param structs)
  are a separate, long-standing pattern here. Leave them alone unless asked.

## 2. No nested or chained ternaries

A single `cond ? a : b` is fine. Choosing among three or more values uses if/else or switch.

```cpp
// WRONG
const int w = a ? 1 : (b ? 2 : 4);

// RIGHT
int w = 4;
if (a) {
    w = 1;
} else if (b) {
    w = 2;
}
```

Losing `const` is acceptable. Use a switch when one discriminant drives all branches.
This also covers ternaries inside a ternary's condition or branches, and `EXPECT_EQ(...)`
arguments that nest them.

## 3. Short comments

- Use the shortest comment that is still correct.
- Do not restate what the code says or what a nearby doc comment already explains.
  Point at the canonical explanation in one line instead ("see CombineLsaMode").
- Explain why, not what. Skip multi-line essays on struct fields and branches.
- Longer design rationale belongs in `docs/`, not inline.

## 4. Braced single-statement bodies, on their own line

Never write one-line control flow, including `goto`, `return`, `continue` and `break`.

```cpp
// WRONG
if (cond) goto Done;
if (x < 0) continue;
for (int i = 0; i < n; ++i) sum += a[i];
if (a) lo = mid + 1; else hi = mid;

// RIGHT
if (cond) {
    goto Done;
}
```

## 5. Indent every nested block

Contents of any `{ }` are indented one level, including bare scoping blocks used to
limit variable lifetime (for example to keep a `goto` from jumping over an initialisation).

```cpp
// WRONG
{
constexpr int kNumElemsPerRead = 4;
}

// RIGHT
{
    constexpr int kNumElemsPerRead = 4;
}
```

Re-check indentation after any edit that adds or moves a brace.

## 6. No inline asm outside `device_primitives.cuh`

All hardware-specific code lives in `nccl_ep/device/device_primitives.cuh`: inline PTX/SASS
(`asm`, `__asm__`), and thin wrappers over fences, scoped or ordered loads, stores and atomics,
mbarriers, async copies, and cache-hint or volatile accesses.

- Before writing any such code, search `device_primitives.cuh` for an existing wrapper and use it.
- If none exists, add the wrapper there (named for what it does, `__device__ __forceinline__`)
  and call it from the kernel. Do not write `asm` at the use site.
- The same applies to a new variant of an existing primitive (another scope, order or width).
  Extend the existing family rather than writing a one-off copy.

```cpp
// WRONG (in a kernel header)
asm volatile("fence.acq_rel.gpu;" ::: "memory");

// RIGHT
nccl_ep::memory_fence_gpu();
```

## 7. Reuse existing helpers

Do not re-implement small utilities that already exist, such as `ceil_div`, `align`, and the
bitmap helpers (`bit_words`, `bit_word`, `test_bit`). These live in `nccl_ep/include/common.hpp`
and are usable from host and device.

- Before writing a helper, grep `include/` and `device/` for the operation.
- Do not hand-roll the arithmetic inline either: `(n + 31) / 32` is `nccl_ep::ceil_div(n, 32)`
  and `((n + a - 1) / a) * a` is `nccl_ep::align(n, a)`.
- If an existing helper is almost right, extend it in place instead of adding a near-copy.

## 8. No code duplication, modularize

- Extract repeated logic into a function or template, instead of copy-pasting a block and
  changing a constant.
- Dtype, width, mode or order variants are template parameters or `if constexpr` branches
  of one implementation, not separate copies.
- When two kernels share a phase (decode, permute, scan, pack), put it in a shared
  `__device__ __forceinline__` helper that both call.
- Before adding code, check whether the same logic already exists elsewhere in the repo.
  Reuse it or move it somewhere shared.

## 9. Keep functions readable

- A function does one thing, and its body fits on a screen. As a guideline, split anything
  much beyond about 60 lines unless it is a flat sequence that reads naturally.
- Deep nesting (more than about three levels of loops and branches) is a signal to extract
  a helper or return early.
- Long kernels are composed from named `__device__ __forceinline__` phases, so the kernel body
  reads as an outline.
- Do not grow an already long function. Extract the new logic into its own function.

## 10. Fixes are new commits

When addressing review feedback on already-committed work, add a new commit on top.
Do not use `git commit --amend`, `--fixup` or `rebase --autosquash` to fold the fix into the
earlier commit, even on an unpushed branch.

- The commit message says what was wrong and why the change is correct.
- Knock-on edits (doc line numbers, tests) go in the same fix commit.
- Fold commits only when explicitly asked.

## Review checklist

When reviewing a diff, grep the added lines for:

- `= ` inside a function parameter list, or a template parameter default
- `?` appearing twice in one expression
- `if (`, `else`, `for (` or `while (` followed by a statement on the same line
- a `{` followed by a line at the same indentation as the brace
- comments longer than about two lines that repeat nearby documentation
- `asm`, `__asm__` or `asm volatile` anywhere outside `device/device_primitives.cuh`
- a local definition or inline expansion of `ceil_div`, `align`, `div_up`, `round_up` or
  `(x + n - 1) / n`
- a newly added block that closely mirrors an existing one, or a function that grew past
  about 60 lines
