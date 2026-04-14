# Experimental Method

A structured approach for investigating problems where the root cause is unclear. Follows the scientific method: observe, hypothesize, test, conclude.

## When to Use

- Performance regressions with no obvious cause
- Unexpected behavior that could have multiple explanations
- Comparing approaches where the tradeoffs aren't clear
- Any problem where "just try it" would waste time without learning

## Phase 1: Hypothesis Generation

**Goal:** Break a large, vague problem into specific, testable claims.

1. **State the problem clearly.** What is observed? What was expected? Include numbers.
   - Bad: "TTFT is slow"
   - Good: "TTFT p50 is 144ms on our tool vs 98ms on InferenceX for identical ISL=1024 random tokens on the same server"

2. **Gather context.** Spin up agents to explore the problem space in parallel:
   - Read relevant code paths
   - Check configs, flags, and server state
   - Review prior results and logs
   - Search for known issues or related work

3. **Generate hypotheses.** Each hypothesis must be:
   - **Specific:** names the exact mechanism ("prefix cache hit rate differs because RNG produces different token distributions")
   - **Falsifiable:** has a clear test that would disprove it
   - **Independent:** can be tested without resolving other hypotheses first

4. **Rank by likelihood and cost to test.** Test cheap, high-probability hypotheses first.

**Output:** A numbered list of hypotheses, each with:
- H1: [claim]
- Why it could be true: [reasoning]
- How to test: [one-line experiment description]
- Expected result if true: [what you'd see]
- Expected result if false: [what you'd see instead]

## Phase 2: Experiment Design

**Goal:** Create a step-by-step plan to test the top hypothesis. Do not start executing until the plan is complete.

1. **Define the control and treatment.**
   - Control: the baseline (current behavior, existing tool, default config)
   - Treatment: the single variable you're changing to test the hypothesis

2. **Isolate variables.** Only change one thing at a time. If you change two things and the result changes, you don't know which one caused it.

3. **Write the execution steps.** For each step:
   - Exact command or code change
   - Expected output
   - What could go wrong and how to handle it
   - How to verify the step succeeded before moving to the next

4. **Define success/failure criteria upfront.**
   - What metric will you measure?
   - What threshold separates "hypothesis confirmed" from "hypothesis rejected"?
   - How many samples do you need for confidence?

5. **Anticipate issues:**
   - Server state: does the server need restarting between runs? (warm cache vs cold)
   - Ordering effects: should you randomize run order?
   - Resource contention: is anything else using the GPU/CPU?
   - Flaky results: what's the variance? Do you need multiple runs?

**Output:** A numbered checklist of steps that can be executed sequentially without further decision-making.

## Phase 3: Execute and Collect Results

**Goal:** Run the experiment exactly as designed. Do not deviate from the plan.

1. **Run the control.** Record all outputs, not just the metric you care about.
2. **Run the treatment.** Same conditions, same recording.
3. **Save raw data.** Results JSON, server logs, GPU utilization — anything that might matter later.
4. **Do not interpret results during collection.** Just record.

**If something unexpected happens during execution:**
- Document what happened
- Do not adjust the experiment mid-run
- Finish the run, then decide whether to re-run or redesign

**Output:** Raw results with full metadata (timestamps, configs, flags, hardware state).

## Phase 4: Analysis and Conclusion

**Goal:** Determine whether the hypothesis is supported, rejected, or inconclusive.

1. **Compare control vs treatment.** Use the success criteria defined in Phase 2.

2. **State the conclusion clearly:**
   - **Supported:** "H1 confirmed — changing X caused Y to improve by Z%. The mechanism is [explanation]."
   - **Rejected:** "H1 rejected — changing X had no effect on Y (control: A, treatment: A). The root cause is elsewhere."
   - **Inconclusive:** "Results are ambiguous (control: A ± B, treatment: C ± D, overlap in confidence intervals). Need more samples or a different test."

3. **Document what you learned** even if the hypothesis was wrong. A rejected hypothesis narrows the search space.

4. **Decide next action:**
   - If confirmed: implement the fix, verify with a production-representative run
   - If rejected: move to the next hypothesis (go back to Phase 2 with H2)
   - If inconclusive: refine the experiment (reduce variance, increase samples, isolate further)

## Example

```
Problem: TTFT is 47% higher on our tool vs InferenceX for identical workloads.

H1: Different RNG produces different token distributions → different prefix cache behavior
H2: We double-wrap the chat template → longer effective prefill
H3: Our arrival pattern (semaphore) vs theirs (rate-limited) causes different queue depths

Phase 2 for H1:
  1. Implement legacy RNG matching InferenceX exactly
  2. Run both tools with same seed, same ISL/OSL
  3. Compare TTFT — if gap closes, H1 confirmed

Result: TTFT gap closed from 47% to 3%. H1 confirmed.
Remaining 3% investigated via H2 → also confirmed (minor effect).
H3 not tested — no longer needed.
```

## Rules

- Never skip Phase 2. "Just trying it" without a plan wastes time and produces uninterpretable results.
- Never change multiple variables at once. You will not know what caused the change.
- Always save raw data. You will want to re-analyze later with different questions.
- A rejected hypothesis is still progress. Document it and move on.
