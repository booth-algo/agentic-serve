# Simulator

`simulator/` now contains only the shape-only vLLM scheduler simulator.

The old kernel-latency/event-loop serving predictor was removed because it had
started to obscure the current experiment. The active path is:

- replay vLLM-like scheduler-step shape,
- compare simulated shape against engine-step traces,
- attach TPOT timing only after scheduler shape is validated.

No H100 command should write outside `/data48/kevinlau/tmp`.
