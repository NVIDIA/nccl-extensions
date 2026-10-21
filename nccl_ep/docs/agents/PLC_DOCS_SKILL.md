---
name: plc-docs
description: "Rules for PLC design documents in nccl-extensions (nccl_ep/docs/plc/): never edit an existing PLC document, write a new one instead, and always prefix its directory with the Merge Request number (ask the user for it if not given). Use whenever creating, updating, or asked to fix a PLC document."
---

# PLC documents

PLC design documents live under `nccl_ep/docs/plc/<dir>/README.md`, one
directory per change, indexed from `nccl_ep/docs/plc/README.md`.

## 1. Never update an existing PLC document

A PLC document, once written, is not edited again — not for a correction, a
follow-up, new benchmark numbers, or review feedback on its own content.

- If asked to update, fix, or add to a PLC document, write a **new** PLC
  document instead and add it to the index in `nccl_ep/docs/plc/README.md`.
- Cross-link from the new document to the one it follows up on (and vice
  versa, if that document is also being added to in the same change) rather
  than rewriting the original.
- This applies to the PLC's own `README.md` content. It does not apply to the
  top-level `nccl_ep/docs/plc/README.md` index, which is expected to grow
  with one new line per PLC.

## 2. Every new PLC directory is prefixed with its MR number

A new PLC directory is named `<MR_NUMBER>-<slug>` (for example
`285-ll-tma-dispatch`), and its `README.md` title ends with `(MR !<MR_NUMBER>)`.

- If the MR number is not already known, ask the user for it before creating
  the directory. Do not guess it, and do not create the directory unnumbered
  and rename it later.
