---
description: Systematic approach to debugging complex integration problems where code runs without errors but produces wrong results
---

# Debugging Complex Problems

Use this workflow when facing problems without a clear cause — where you need to read code, investigate, and iterate many times to make progress. Particularly useful when code runs without errors but produces wrong results.

## Phase 1: Research Before Changing Code

Before writing or modifying any code, exhaust external knowledge first. Most problems have been solved before.

1. **Search for reference implementations.** Find how other projects solve the same problem. Read their source code. Prioritize battle-tested, popular implementations over documentation alone.
2. **Read the data's own metadata.** Many formats, protocols, and systems are self-describing. Inspect embedded configs, schemas, headers, version fields, and metadata before making assumptions about structure.
3. **Search for known issues.** Check GitHub issues, Stack Overflow, forum posts, changelogs, and READMEs for known gotchas, breaking changes, or migration notes related to the specific versions you're using.

> **Why this matters:** Hours of code investigation can be shortcut by 15 minutes of reading how someone else already solved the same problem.

## Phase 2: Measure Before Theorizing

When the cause isn't obvious, collect quantitative data before forming hypotheses.

1. **Add diagnostic instrumentation.** Write small, targeted scripts or logging that measure what's actually happening — don't guess. Compare expected vs actual at every boundary.
2. **Verify data integrity.** Check counts, shapes, types, and value ranges at each transformation stage. Confirm that inputs to each component match what that component expects.
3. **Identify what's different.** If something used to work or works in a reference implementation, systematically compare: configs, versions, data, environment, call sequences. Narrow the delta.

> **Why this matters:** A 5-line diagnostic script is worth more than an hour of reading code and reasoning about what *might* be wrong.

## Phase 3: Find the Root Cause, Not the Symptom

When encountering errors or unexpected behavior during investigation:

1. **Ask "WHY is this wrong?" before fixing.** Don't patch individual errors. Multiple related errors usually share a single root cause (wrong config, missing step, version mismatch, wrong assumptions).
2. **Trace backwards from the failure.** Follow the data path from the point of failure back to its source. Where does the wrong value first appear?
3. **Beware symptom-fixing loops.** If you've fixed 3+ related errors and new ones keep appearing, stop. You're likely treating symptoms while the root cause is upstream. Step back and re-examine your assumptions.

## Phase 4: Treat Silent Failures as the Most Dangerous

The hardest bugs are the ones that don't throw errors.

1. **Audit lenient/permissive operations.** `strict=False`, `try/except pass`, `ignore_missing=True`, default fallbacks, optional parameters — these all hide mismatches. Add monitoring or assertions whenever using them.
2. **"It runs" ≠ "It works."** Define concrete success criteria beyond "no exceptions." e.g., "0 unexpected drops, 0 unexpected misses", "output matches reference within tolerance", "all expected components are present."
3. **Log what was *skipped*, not just what was *done*.** The absence of action is often the bug.

## Anti-Patterns to Avoid

| Anti-Pattern | Better Approach |
| --- | --- |
| Reverse-engineering from scratch | Find and study the reference implementation |
| Hardcoding values by reading source | Read config/metadata from the data itself |
| Fixing errors one-by-one as they appear | Find why they're wrong in the first place |
| "It runs without errors = it works" | Verify outputs quantitatively |
| Hours of theory before any measurement | Write a small diagnostic script first |
| Assuming you know the correct config | Check what the system/data says the config is |
| Deep-diving into one theory for too long | Time-box investigations; pivot if no progress |

## When to Restart Your Approach

If you've been iterating for more than 3 cycles without clear progress:

1. **List your assumptions.** Write down everything you're assuming is correct. Challenge each one.
2. **Go back to Phase 1.** Search for new reference implementations or documentation you might have missed.
3. **Simplify the reproduction.** Strip the problem down to the smallest possible case that still fails. Remove all variables you can.
4. **Explain the problem to someone (or write it out).** Articulating the problem often reveals the gap in your reasoning.
