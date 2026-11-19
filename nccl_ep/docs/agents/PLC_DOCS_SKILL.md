---
name: plc-docs
description: "Rules for PLC design documents in nccl-extensions (nccl_ep/docs/plc/): not every MR needs one (advise a PLC for API changes and significant algorithmic or infrastructure changes, and ask the user explicitly whether to add it), a new change gets a new PLC document, a fix to a past PLC is made in that document and called out in the MR, and always prefix its directory with the Merge Request number (ask the user for it if not given). Use whenever creating, updating, or asked to fix a PLC document, and when finishing a change that may warrant one."
---

# PLC documents

PLC design documents live under `nccl_ep/docs/plc/<dir>/README.md`, one
directory per change, indexed from `nccl_ep/docs/plc/README.md`.

## 1. When a PLC is warranted

A PLC is not required for every MR. Bug fixes, refactors without a behavior
change, tests, tuning of an existing constant, build fixes and documentation
edits do not need one.

Advise a PLC when the change contains any of:

- an API change: a new or changed public function, struct, enum, config
  field, environment variable or documented contract;
- a significant algorithmic change: a new kernel or protocol, a new wire or
  buffer layout, a new mode or dispatch policy, or a changed synchronization
  scheme;
- a significant infrastructure change: the JIT, the build, the benchmark or
  test harness, or the review tooling, in a way other contributors must know.

Do not decide alone in either direction. When one of these is present and the
change has no PLC, ask the user explicitly whether to add one, naming what in
the change prompted the question. Do not create a PLC unasked, and do not
skip the question because the change looks small.

## 2. New change, new document; a fix to an old document goes into that document

A PLC document describes one change. What a later MR does with it depends on
why the MR touches it.

- **The MR fixes the past PLC itself**: a wrong statement, a broken link, a
  figure or number that was incorrect when written, or a description of code
  that this MR corrects. Edit the existing document in place. Do not create a
  new PLC directory for a correction.
- **The MR adds a new change** (a feature, a redesign, a follow-up
  optimization, new benchmark results for new code) and a PLC is wanted
  (section 1): write a **new** PLC document, add it to the index in
  `nccl_ep/docs/plc/README.md`, and cross-link the two. Do not rewrite the
  older document to describe the new design.
- An edit to an existing PLC is always called out: list the edited document
  and the reason in the MR description, so a reviewer sees it was intended.
  Keep the edit to what the fix needs; do not restyle or reorganize the rest.
- The top-level `nccl_ep/docs/plc/README.md` index grows with one new line per
  PLC; an existing entry changes only when its document's fix requires it.

## 3. Every new PLC directory is prefixed with its MR number

A new PLC directory is named `<MR_NUMBER>-<slug>` (for example
`285-ll-tma-dispatch`), and its `README.md` title ends with `(MR !<MR_NUMBER>)`.

- If the MR number is not already known, ask the user for it before creating
  the directory. Do not guess it, and do not create the directory unnumbered
  and rename it later.
