# gen-commit-msg

Generate a conventional commit message for staged changes.

## Steps

1. Run `git diff --staged` to see all staged changes
2. Run `git log --oneline -10` to see recent commit style
3. Categorize the change:
   - `feat`: new profile, new mode, new engine support
   - `fix`: bug fix in metrics, client, or runner
   - `bench`: new benchmark results or result updates
   - `docs`: markdown/ updates, README, comments
   - `refactor`: restructuring without behavior change
   - `chore`: deps, config, scripts
4. Generate message format:
   ```
   <type>(<scope>): <short description>

   <body explaining what and why, not how>
   ```

## Scope examples

- `runner`, `metrics`, `client` — core benchmark code
- `modes` — stress-test/single-turn/multi-turn mode changes
- `profiles` — workload profile changes
- `engines` — vllm/sglang/trtllm engine changes
- `results` — benchmark result updates
- `scripts` — launch_server.sh, bench.sh changes
