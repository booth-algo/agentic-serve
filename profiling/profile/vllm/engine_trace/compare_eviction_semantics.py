"""S7/S8/S9 adjudication: sim eviction cluster vs real vLLM v1 engine traces.

ANALYSIS ONLY — no production code paths import this. Engine traces are
VALIDATION ORACLES, never predictor inputs.

Inputs: the L4 full-cell captures (`l4_engine_trace_*_full_cell.jsonl` +
`*_token_history.json`). The jsonl carries per-scheduler-step
`engine_cache_truth.kv_events`: `get_computed_blocks` (LIVE computed_tokens at
scheduling time — the S8 oracle) and `allocate_slots` (a complete allocation
log incl. decode growth), plus queue contents and `preempted_request_ids`.
The token_history carries exact prompt/output token ids per (session, turn),
which lets us run a faithful re-implementation of the vLLM v1 BlockPool
(block-hash prefix cache + LRU free queue, tail-first frees) and VALIDATE it
against the trace's own computed_tokens — once validated, the replica exposes
state the trace can't show directly (residency at the turn barrier, the iden-
tity/status of every eviction victim).

Adjudicated questions:
  S7  eviction ORDER + GRANULARITY: engine = free-queue LRU-oldest,
      block-by-block (partial, tail-first). Sim tier-2 = whole-session,
      MRU-first ('tail' default). Quantified via (a) full/partial/zero-hit
      classification of every turn>0 lookup, (b) per-turn correlation between
      a session's finish order at turn t-1 and its prefix loss at turn t.
  S8  hit/miss FROZEN at barrier vs LIVE at scheduling: replica snapshot at
      herd release vs the trace's live computed_tokens — the phantom-hit
      tokens the freeze grants.
  S9  herd_pending protection: status of every evicted block's owner at
      eviction time (waiting herd member / in-flight / finished-this-turn /
      drained-dead). The sim forbids evicting waiting herd members; the
      engine has no such concept.

Counterfactual replay: the production ``PrefixLRUCache`` (session-granular)
driven by the engine's own admission/finish order under the sim's rule
variants (frozen vs live x tail-whole / lru-whole / lru-partial), re-prefill
tokens per turn compared against engine truth.

Usage:
  python3 -m profiling.profile.vllm.engine_trace.compare_eviction_semantics \
      --trace-dir profile_data/_archive/l4_queue_trace_run
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

BLOCK = 16

# --------------------------------------------------------------------- loading


def _ids(text: str | None) -> list[str]:
    return [t for t in (text or "").split() if t]


def load_steps(jsonl_path: Path) -> list[dict]:
    steps = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ct = d.get("engine_cache_truth")
        d["_kv_events"] = json.loads(ct).get("kv_events", []) if ct else []
        d["_running"] = _ids(d.get("running_request_ids"))
        d["_scheduled"] = _ids(d.get("scheduled_request_ids"))
        d["_preempted"] = _ids(d.get("preempted_request_ids"))
        steps.append(d)
    return steps


def load_history(hist_path: Path) -> tuple[dict[int, dict], dict[tuple[int, int], dict]]:
    """Returns (global_request_counter -> entry, (session,turn) -> entry).

    The engine assigns request ids '<n>-<hash>' with n a GLOBAL sequential
    counter across the run; submission order is turn-major, session-ascending
    within the turn (verified: 100% prompt-token match on every L4 trace)."""
    per_req = json.load(open(hist_path))["per_request"]
    order = sorted(per_req, key=lambda e: (e["turn_index"], e["session_id"]))
    n2e = {i: e for i, e in enumerate(order)}
    st2e = {(e["session_id"], e["turn_index"]): e for e in per_req}
    return n2e, st2e


def req_n(request_id: str) -> int:
    return int(request_id.split("-")[0])


# ------------------------------------------------- vLLM v1 BlockPool replica


class EnginePoolReplica:
    """Faithful re-implementation of vLLM v1 BlockPool + prefix-cache semantics.

    * block hash chain over token ids (parent-keyed), full blocks only;
    * single free queue holding every refcount-0 block, initial order = block
      ids; eviction = popleft (LRU-OLDEST), unregisters that block's hash;
    * cache hit revives blocks (removes from free queue, ref++);
    * a finished/preempted request's blocks are freed in REVERSE order (tail
      first) so tail blocks are evicted before head-of-prefix;
    * lookup is capped at (num_tokens - 1) // BLOCK full blocks (vLLM never
      returns the entire prompt as cached).
    Validated against the trace's own get_computed_blocks computed_tokens.
    """

    def __init__(self, capacity_blocks: int, evict_order: str = "lru") -> None:
        self.evict_order = evict_order  # 'lru' = popleft oldest (vLLM); 'mru' = pop newest (falsification)
        self.capacity = capacity_blocks
        self.free_q: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(capacity_blocks)
        )
        self.hash_of: dict[int, tuple | None] = {}      # block_id -> hash key
        self.by_hash: dict[tuple, int] = {}             # hash key -> block_id
        self.ref: dict[int, int] = defaultdict(int)
        self.owner: dict[int, tuple[int, int]] = {}     # block_id -> (sid, blk_idx) last writer
        self.req_blocks: dict[str, list[int]] = {}      # rid -> owned block ids
        self.req_computed: dict[str, int] = {}
        self.evict_log: list[dict] = []                 # one entry per evicted block

    # -- hashing
    @staticmethod
    def chain(tokens: list[int]) -> list[tuple]:
        keys, parent = [], None
        for i in range(len(tokens) // BLOCK):
            key = (parent, tuple(tokens[i * BLOCK:(i + 1) * BLOCK]))
            keys.append(key)
            parent = key
        return keys

    def lookup(self, keys: list[tuple], num_tokens: int) -> int:
        """get_computed_blocks: longest cached prefix, in BLOCKS (capped)."""
        cap = max(0, (num_tokens - 1) // BLOCK)
        n = 0
        for key in keys[:cap]:
            if key in self.by_hash:
                n += 1
            else:
                break
        return n

    def free_count(self) -> int:
        return len(self.free_q)

    def _evict_one(self, ctx: dict) -> int:
        bid, _ = self.free_q.popitem(last=(self.evict_order == "mru"))  # vLLM: LRU-OLDEST
        key = self.hash_of.get(bid)
        if key is not None:
            owner = self.owner.get(bid)
            self.evict_log.append({**ctx, "owner": owner})
            self.by_hash.pop(key, None)
            self.hash_of[bid] = None
        return bid

    def allocate(self, rid: str, sid: int, keys: list[tuple], hit_blocks: int,
                 total_blocks_needed: int, ctx: dict) -> bool:
        """First allocation for a (re)scheduled request: revive hits, evict for the rest."""
        blocks: list[int] = []
        for i in range(hit_blocks):
            bid = self.by_hash[keys[i]]
            if self.ref[bid] == 0:
                self.free_q.pop(bid, None)
            self.ref[bid] += 1
            blocks.append(bid)
        need = total_blocks_needed - hit_blocks
        if need > len(self.free_q):
            return False
        for _ in range(need):
            bid = self._evict_one(ctx)
            self.ref[bid] += 1
            blocks.append(bid)
        self.req_blocks[rid] = blocks
        return True

    def grow(self, rid: str, total_blocks_needed: int, ctx: dict) -> bool:
        blocks = self.req_blocks.setdefault(rid, [])
        need = total_blocks_needed - len(blocks)
        if need <= 0:
            return True
        if need > len(self.free_q):
            return False
        for _ in range(need):
            bid = self._evict_one(ctx)
            self.ref[bid] += 1
            blocks.append(bid)
        return True

    def register_full_blocks(self, rid: str, sid: int, keys: list[tuple], computed_tokens: int) -> None:
        """cache_full_blocks: register hashes for blocks fully computed."""
        blocks = self.req_blocks.get(rid, [])
        for i in range(min(computed_tokens // BLOCK, len(blocks), len(keys))):
            bid = blocks[i]
            if self.hash_of.get(bid) is None:
                old = self.by_hash.get(keys[i])
                if old is not None and old != bid:
                    continue  # another block already holds this hash
                self.hash_of[bid] = keys[i]
                self.by_hash[keys[i]] = bid
                self.owner[bid] = (sid, i)

    def free_request(self, rid: str) -> None:
        """Finished/preempted: blocks to the free queue tail-FIRST (reverse order)."""
        for bid in reversed(self.req_blocks.pop(rid, [])):
            self.ref[bid] -= 1
            if self.ref[bid] == 0:
                self.free_q[bid] = None
        self.req_computed.pop(rid, None)

    def session_resident_blocks(self, keys: list[tuple]) -> int:
        """Residency snapshot (same walk as lookup, uncapped) — the barrier oracle."""
        n = 0
        for key in keys:
            if key in self.by_hash:
                n += 1
            else:
                break
        return n


# ------------------------------------------------------------- trace replay


def replay_engine(steps: list[dict], n2e: dict[int, dict], st2e: dict,
                  evict_order: str = "lru") -> dict:
    """Drive the replica with the trace's own event stream; validate lookups;
    log per-lookup (live truth, replica prediction, barrier residency) and the
    eviction attribution."""
    # capacity: free blocks at first record + blocks already allocated in it
    first = steps[0]
    capacity = int(first["free_kv_blocks"]) + int(first.get("engine_new_block_count") or 0)
    pool = EnginePoolReplica(capacity, evict_order=evict_order)

    keys_cache: dict[int, list[tuple]] = {}
    full_seq: dict[int, list[int]] = {}

    def keys_for(n: int) -> list[tuple]:
        if n not in keys_cache:
            e = n2e[n]
            full_seq[n] = list(e["prompt_token_ids"]) + list(e["output_token_ids"])
            keys_cache[n] = EnginePoolReplica.chain(full_seq[n])
        return keys_cache[n]

    # finish step per request: last step where it appears scheduled or running
    last_seen: dict[str, int] = {}
    for i, s in enumerate(steps):
        for rid in set(s["_running"]) | set(s["_scheduled"]):
            last_seen[rid] = i

    lookups: list[dict] = []          # one per get_computed_blocks event
    barrier_snap: dict[tuple[int, int], int] = {}   # (sid, turn) -> resident blocks at herd release
    seen_turn = -1
    herd_first_lookup: dict[str, int] = {}
    in_flight: set[str] = set()
    finished_this_turn: set[int] = set()
    turn_of_sid_alive: dict[int, int] = {}

    validate_ok = 0
    validate_bad: list[tuple] = []

    for i, s in enumerate(steps):
        turn = int(s["turn_index"])
        if turn != seen_turn:
            # herd release barrier: snapshot every arriving session's residency
            seen_turn = turn
            finished_this_turn = set()
            for (sid, t), e in st2e.items():
                if t == turn:
                    n = [k for k, v in n2e.items() if v is e][0]
                    barrier_snap[(sid, turn)] = pool.session_resident_blocks(keys_for(n))
        for ev in s["_kv_events"]:
            rid = ev["request_id"]
            n = req_n(rid)
            e = n2e[n]
            sid, t = e["session_id"], e["turn_index"]
            keys = keys_for(n)
            if ev["event"] == "get_computed_blocks":
                pred = pool.lookup(keys, int(ev["request_num_tokens"]))
                truth = int(ev["computed_tokens"]) // BLOCK
                if pred == truth:
                    validate_ok += 1
                else:
                    validate_bad.append((i, rid, pred, truth))
                herd_first_lookup.setdefault(rid, i)
                lookups.append({
                    "step": i, "rid": rid, "sid": sid, "turn": t,
                    "prompt": int(ev["request_num_prompt_tokens"]),
                    "live_blocks": truth,
                    "replica_blocks": pred,
                    "barrier_blocks": barrier_snap.get((sid, t), 0),
                })
            else:  # allocate_slots
                if ev.get("allocation_failed"):
                    continue
                committed = int(ev["request_num_computed_tokens"]) + 0  # before exec
                new_toks = int(ev["num_new_tokens"])
                total_tokens = min(int(ev["request_num_tokens"]), len(full_seq[n]))
                total_blocks = -(-min(committed + new_toks, total_tokens) // BLOCK)
                ctx = {"step": i, "turn": turn, "for_rid": rid, "for_sid": sid}
                if rid not in pool.req_blocks:
                    hit = pool.lookup(keys, int(ev["request_num_tokens"]))
                    # vLLM reports computed BEFORE new tokens for fresh scheds
                    pool.allocate(rid, sid, keys, hit, total_blocks, ctx)
                    in_flight.add(rid)
                else:
                    pool.grow(rid, total_blocks, ctx)
                pool.register_full_blocks(rid, sid, keys, min(committed + new_toks, total_tokens))
        # end-of-step: preemptions then finishes free blocks before next schedule
        for rid in s["_preempted"]:
            if rid in pool.req_blocks:
                pool.free_request(rid)
                in_flight.discard(rid)
        # deterministic free order among same-step finishers: engine processes
        # finished requests in running-list order; req counter order matches it.
        for rid in sorted(in_flight, key=req_n):
            if last_seen.get(rid, -1) <= i:
                pool.free_request(rid)
                in_flight.discard(rid)
                finished_this_turn.add(n2e[req_n(rid)]["session_id"])

    return {
        "capacity": capacity,
        "lookups": lookups,
        "barrier_snap": barrier_snap,
        "evict_log": pool.evict_log,
        "validate_ok": validate_ok,
        "validate_bad": validate_bad,
        "last_seen": last_seen,
    }


# ----------------------------------------------------------------- analyses


def lcp_tokens(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def effective_lookups(lookups: list[dict]) -> dict[tuple[int, int], dict]:
    """Last lookup per (sid, turn) = the one the scheduler acted on."""
    out: dict[tuple[int, int], dict] = {}
    for lk in lookups:
        out[(lk["sid"], lk["turn"])] = lk
    return out


def s7_granularity(eff: dict, st2e: dict) -> dict:
    """Classify every turn>0 effective lookup: full hit / PARTIAL loss / zero.

    expect = block-quantized LCP(prompt_t, prompt_{t-1}+output_{t-1}), capped
    at (prompt-1)//BLOCK — the engine's own reuse ceiling."""
    cls = {"full": 0, "partial": 0, "zero": 0}
    loss_fracs: list[float] = []
    per_turn: dict[int, dict] = defaultdict(lambda: {"full": 0, "partial": 0, "zero": 0,
                                                     "expect_tok": 0, "live_tok": 0})
    for (sid, t), lk in sorted(eff.items()):
        if t == 0:
            continue
        prev = st2e.get((sid, t - 1))
        cur = st2e[(sid, t)]
        if prev is None:
            continue
        hist = list(prev["prompt_token_ids"]) + list(prev["output_token_ids"])
        reuse = lcp_tokens(hist, list(cur["prompt_token_ids"]))
        expect = min(reuse // BLOCK, (lk["prompt"] - 1) // BLOCK)
        live = lk["live_blocks"]
        pt = per_turn[t]
        pt["expect_tok"] += expect * BLOCK
        pt["live_tok"] += live * BLOCK
        if expect <= 0:
            continue
        if live >= expect:
            cls["full"] += 1; pt["full"] += 1
        elif live == 0:
            cls["zero"] += 1; pt["zero"] += 1
            loss_fracs.append(1.0)
        else:
            cls["partial"] += 1; pt["partial"] += 1
            loss_fracs.append(1.0 - live / expect)
    return {"cls": cls, "loss_fracs": loss_fracs, "per_turn": dict(per_turn)}


def s7_order(eff: dict, st2e: dict, last_seen: dict, n_of: dict) -> list[tuple[int, float, int]]:
    """Per turn: Spearman-ish correlation between a session's FINISH ORDER at
    turn t-1 (earlier = its blocks freed earlier = LRU-older in the engine
    free queue) and its prefix LOSS fraction at turn t.

    Engine LRU-oldest predicts POSITIVE-loss-on-early-finishers (negative corr
    of finish_rank vs loss). Sim 'tail' (MRU-first preemption) predicts the
    opposite sign."""
    res = []
    turns = sorted({t for (_s, t) in eff if t > 0})
    for t in turns:
        rows = []
        for (sid, tt), lk in eff.items():
            if tt != t or (sid, t - 1) not in eff:
                continue
            prev = st2e[(sid, t - 1)]
            hist = list(prev["prompt_token_ids"]) + list(prev["output_token_ids"])
            reuse = lcp_tokens(hist, list(st2e[(sid, t)]["prompt_token_ids"]))
            expect = min(reuse // BLOCK, (lk["prompt"] - 1) // BLOCK)
            if expect <= 0:
                continue
            loss = 1.0 - min(1.0, lk["live_blocks"] / expect)
            rows.append((sid, loss))
        # finish order: last_seen step of the previous-turn request
        ranked = []
        for sid, loss in rows:
            prev_lk = eff[(sid, t - 1)]
            rid = prev_lk["rid"]
            ranked.append((last_seen.get(rid, 0), loss))
        lossy = [x for x in ranked if x[1] > 0]
        if len(lossy) < 3 or len(set(x[0] for x in ranked)) < 3:
            continue
        # rank correlation (Spearman via rank transform, ties = mean rank)
        def ranks(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            r = [0.0] * len(vals)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg = (i + j) / 2.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        fs = ranks([x[0] for x in ranked])
        ls = ranks([x[1] for x in ranked])
        mf, ml = statistics.fmean(fs), statistics.fmean(ls)
        num = sum((a - mf) * (b - ml) for a, b in zip(fs, ls))
        den = (sum((a - mf) ** 2 for a in fs) * sum((b - ml) ** 2 for b in ls)) ** 0.5
        if den > 0:
            res.append((t, num / den, len(lossy)))
    return res


def s8_frozen_vs_live(eff: dict, barrier_snap: dict) -> dict:
    """Phantom-hit tokens the barrier freeze grants: frozen(snapshot-at-release,
    capped at live ceiling) minus live computed, per turn>0 request."""
    per_turn = defaultdict(lambda: {"frozen_tok": 0, "live_tok": 0, "reqs": 0,
                                    "flips": 0})
    for (sid, t), lk in sorted(eff.items()):
        if t == 0:
            continue
        frozen = min(barrier_snap.get((sid, t), 0), (lk["prompt"] - 1) // BLOCK)
        live = lk["live_blocks"]
        pt = per_turn[t]
        pt["frozen_tok"] += frozen * BLOCK
        pt["live_tok"] += live * BLOCK
        pt["reqs"] += 1
        if frozen > 0 and live < frozen:
            pt["flips"] += 1
    return dict(per_turn)


def s9_victim_status(evict_log: list[dict], eff: dict, st2e: dict,
                     herd_sched_step: dict) -> dict:
    """Status of each evicted block's OWNER session at eviction time:
    waiting-herd-member (sim forbids), already-ran-this-turn, in-flight, or
    not-in-this-herd (dead/drained). Owner = last session whose request wrote
    the block (replica attribution)."""
    counts = defaultdict(int)
    for ev in evict_log:
        owner = ev.get("owner")
        if owner is None:
            counts["untracked"] += 1
            continue
        osid = owner[0]
        turn = ev["turn"]
        key = (osid, turn)
        if key not in st2e:
            counts["dead_or_drained"] += 1
        else:
            sched = herd_sched_step.get(key)
            if sched is None or ev["step"] < sched:
                counts["WAITING_HERD_MEMBER"] += 1     # sim's herd_pending forbids this
            elif ev["step"] > eff[key]["step"]:
                counts["ran_or_running_this_turn"] += 1
            else:
                counts["in_flight_now"] += 1
    return dict(counts)


# --------------------------------------------- counterfactual sim-rule replay


def sim_rule_replay(st2e: dict, eff: dict, capacity: int, last_seen: dict,
                    policy: str, whole: bool, frozen: bool) -> dict:
    """Drive the production PrefixLRUCache (session-granular) with the engine's
    own admission order (effective-lookup step) and finish order (last_seen),
    under a sim rule variant. Emits per-turn re-prefill tokens to compare with
    engine truth. cached_t := LCP(prompt_t, prev ctx) (block-quantized) so the
    workload definition matches the engine-truth `expect`."""
    import simulator.ttft_queue_sim as tqs

    cache = tqs.PrefixLRUCache(capacity, BLOCK)
    # monkeypatch the tier-2 trim granularity for the 'partial' variant
    orig_evict = cache._evict

    def evict_partial(need, hard, soft, policy_=policy):
        if need <= cache.free():
            return True
        free_residents = sorted(
            (s for s in cache.cached if s not in hard and s not in soft and cache.cached[s] > 0),
            key=lambda s: (cache.recency.get(s, -1), s))
        cache._trim_tail(free_residents, need)
        if cache.free() >= need:
            return True
        soft_l = [s for s in cache.cached if s in soft and s not in hard and cache.cached[s] > 0]
        soft_l.sort(key=lambda s: (cache.recency.get(s, -1), s), reverse=(policy_ == "tail"))
        cache._trim_tail(soft_l, need, whole=whole)
        return cache.free() >= need

    cache._evict = evict_partial  # type: ignore[method-assign]

    turns = sorted({t for (_s, t) in st2e})
    per_turn = {}
    for t in turns:
        herd = sorted([sid for (sid, tt) in st2e if tt == t],
                      key=lambda sid: eff[(sid, t)]["step"] if (sid, t) in eff else 1 << 30)
        snap = {sid: cache.cached_blocks(sid) for sid in herd}
        herd_pending = set(herd)
        # interleave admissions (by eff step) and finishes (by last_seen of rid)
        events = []
        for sid in herd:
            lk = eff.get((sid, t))
            if lk is None:
                continue
            events.append((lk["step"], 0, "admit", sid))
            events.append((last_seen.get(lk["rid"], lk["step"]), 1, "finish", sid))
        events.sort()
        in_flight: set[int] = set()
        reprefill_tok = 0
        fa3_proxy = 0.0  # sum M*(R + M/2): the sim's quadratic attention re-encode driver
        miss_cls = {"full": 0, "partial": 0, "zero": 0}
        for _step, _o, kind, sid in events:
            e = st2e[(sid, t)]
            prompt_blocks = -(-len(e["prompt_token_ids"]) // BLOCK)
            if kind == "admit":
                if t > 0 and (sid, t - 1) in st2e:
                    prev = st2e[(sid, t - 1)]
                    hist = list(prev["prompt_token_ids"]) + list(prev["output_token_ids"])
                    cached_tok = lcp_tokens(hist, list(e["prompt_token_ids"]))
                else:
                    cached_tok = 0
                resident_blocks = snap[sid] if frozen else cache.cached_blocks(sid)
                resident = min(cached_tok, resident_blocks * BLOCK)
                reprefill = (cached_tok - resident) + (len(e["prompt_token_ids"]) - cached_tok)
                reprefill_tok += reprefill
                fa3_proxy += reprefill * (resident + 0.5 * reprefill)
                expect = cached_tok // BLOCK
                if expect > 0:
                    if resident // BLOCK >= expect:
                        miss_cls["full"] += 1
                    elif resident == 0:
                        miss_cls["zero"] += 1
                    else:
                        miss_cls["partial"] += 1
                cache.grow_to(sid, prompt_blocks, in_flight | {sid},
                              herd_pending - {sid}, policy)
                in_flight.add(sid)
            else:
                total_blocks = -(-(len(e["prompt_token_ids"]) + len(e["output_token_ids"])) // BLOCK)
                cache.grow_to(sid, total_blocks, in_flight, herd_pending - in_flight, policy)
                in_flight.discard(sid)
                herd_pending.discard(sid)
                cache.touch(sid)
        per_turn[t] = {"reprefill_tok": reprefill_tok, "fa3_proxy": fa3_proxy, **miss_cls}
    return per_turn


def engine_truth_per_turn(eff: dict) -> dict:
    out = defaultdict(int)
    for (sid, t), lk in eff.items():
        out[t] += lk["prompt"] - lk["live_blocks"] * BLOCK
    return dict(out)


# ----------------------------------------------------------------------- main


def analyze_trace(stem: Path) -> None:
    jsonl = stem.with_suffix(".jsonl")
    hist_path = Path(str(stem).replace("_full_cell", "_token_history") + ".json")
    print(f"\n{'=' * 78}\nTRACE {stem.name}")
    steps = load_steps(jsonl)
    n2e, st2e = load_history(hist_path)
    rep = replay_engine(steps, n2e, st2e)
    nlk = rep["validate_ok"] + len(rep["validate_bad"])
    print(f"replica validation: {rep['validate_ok']}/{nlk} lookups exact "
          f"({100.0 * rep['validate_ok'] / max(1, nlk):.2f}%)  capacity={rep['capacity']} blocks")
    if rep["validate_bad"][:5]:
        print("  first mismatches:", rep["validate_bad"][:5])
    # ORDER falsification: same replica, MRU (newest-first) eviction
    rep_mru = replay_engine(steps, n2e, st2e, evict_order="mru")
    nmru = rep_mru["validate_ok"] + len(rep_mru["validate_bad"])
    err_tok_lru = sum(abs(p - t) * BLOCK for (_i, _r, p, t) in rep["validate_bad"])
    err_tok_mru = sum(abs(p - t) * BLOCK for (_i, _r, p, t) in rep_mru["validate_bad"])
    print(f"ORDER falsification — MRU-evicting replica: {rep_mru['validate_ok']}/{nmru} exact "
          f"({100.0 * rep_mru['validate_ok'] / max(1, nmru):.2f}%); "
          f"abs token error LRU={err_tok_lru} vs MRU={err_tok_mru}")

    eff = effective_lookups(rep["lookups"])
    n_of = {}

    # S7 granularity
    g = s7_granularity(eff, st2e)
    cls = g["cls"]
    aff = cls["partial"] + cls["zero"]
    print(f"\nS7 GRANULARITY (turn>0 effective lookups): full-hit={cls['full']} "
          f"partial-loss={cls['partial']} zero-hit={cls['zero']}")
    if aff:
        print(f"  affected lookups that are PARTIAL (engine block-LRU signature): "
              f"{cls['partial']}/{aff} = {100.0 * cls['partial'] / aff:.1f}%")
    if g["loss_fracs"]:
        lf = sorted(g["loss_fracs"])
        med = lf[len(lf) // 2]
        print(f"  loss fraction among lossy: median={med:.3f} "
              f"mean={statistics.fmean(lf):.3f} (sim whole-session would force 1.0)")

    # S7 order
    corr = s7_order(eff, st2e, rep["last_seen"], n_of)
    if corr:
        cs = [c for (_t, c, _n) in corr]
        print(f"\nS7 ORDER: per-turn Spearman(finish-rank@t-1, loss@t) over {len(corr)} "
              f"turns with >=3 lossy sessions: median={statistics.median(cs):+.3f} "
              f"mean={statistics.fmean(cs):+.3f}")
        print("  (NEGATIVE = early finishers lose more = engine free-queue LRU; "
              "sim 'tail' MRU-first predicts POSITIVE)")
        for t, c, n in corr:
            print(f"    turn {t:>2}: corr={c:+.3f} lossy={n}")

    # S8 frozen vs live
    s8 = s8_frozen_vs_live(eff, rep["barrier_snap"])
    tot_frozen = sum(v["frozen_tok"] for v in s8.values())
    tot_live = sum(v["live_tok"] for v in s8.values())
    tot_flips = sum(v["flips"] for v in s8.values())
    tot_reqs = sum(v["reqs"] for v in s8.values())
    print(f"\nS8 FROZEN-vs-LIVE: barrier-frozen credit {tot_frozen} tok vs live "
          f"{tot_live} tok -> freeze over-credits {tot_frozen - tot_live} tok "
          f"({100.0 * (tot_frozen - tot_live) / max(1, tot_frozen):.1f}% of frozen credit) "
          f"across {tot_reqs} turn>0 reqs; {tot_flips} reqs lost prefix MID-TURN")
    worst = sorted(s8.items(), key=lambda kv: kv[1]["frozen_tok"] - kv[1]["live_tok"], reverse=True)[:6]
    for t, v in worst:
        if v["frozen_tok"] - v["live_tok"] > 0:
            print(f"    turn {t:>2}: frozen={v['frozen_tok']} live={v['live_tok']} "
                  f"delta={v['frozen_tok'] - v['live_tok']} flips={v['flips']}/{v['reqs']}")

    # S9 victim status
    herd_sched_step = {k: lk["step"] for k, lk in eff.items()}
    s9 = s9_victim_status(rep["evict_log"], eff, st2e, herd_sched_step)
    tot_ev = sum(s9.values())
    print(f"\nS9 VICTIM STATUS at eviction time ({tot_ev} evicted blocks):")
    for k in sorted(s9, key=s9.get, reverse=True):
        print(f"    {k:<26} {s9[k]:>7}  ({100.0 * s9[k] / max(1, tot_ev):.1f}%)")
    print("  (sim herd_pending forbids WAITING_HERD_MEMBER evictions entirely)")

    # counterfactual sim-rule replay
    truth = engine_truth_per_turn(eff)
    print("\nCOUNTERFACTUAL re-prefill tokens per run (engine admission/finish order, "
          f"pool={rep['capacity']} blocks):")
    total_truth = sum(truth.values())
    print(f"    ENGINE TRUTH (live lookups)           : {total_truth:>9} tok")
    variants = [
        ("sim CURRENT  (frozen + tail  + whole)", "tail", True, True),
        ("sim alt      (frozen + lru   + whole)", "lru", True, True),
        ("sim alt      (frozen + lru   + partial)", "lru", False, True),
        ("sim alt      (live   + tail  + whole)", "tail", True, False),
        ("sim alt      (live   + lru   + partial)", "lru", False, False),
    ]
    var_rows = {}
    for label, pol, whole, frozen in variants:
        pt = sim_rule_replay(st2e, eff, rep["capacity"], rep["last_seen"], pol, whole, frozen)
        tot = sum(v["reprefill_tok"] for v in pt.values())
        var_rows[label] = pt
        print(f"    {label}: {tot:>9} tok  ({100.0 * (tot - total_truth) / max(1, total_truth):+.1f}% vs truth)")
    # per-turn detail for the current rule vs truth vs best alt
    print("    per-turn (truth | current | lru+partial+live):")
    cur = var_rows[variants[0][0]]
    alt = var_rows[variants[4][0]]
    for t in sorted(truth):
        print(f"      t{t:>2}: {truth[t]:>8} | {cur.get(t, {}).get('reprefill_tok', 0):>8} "
              f"| {alt.get(t, {}).get('reprefill_tok', 0):>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="profile_data/_archive/l4_queue_trace_run")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    d = Path(args.trace_dir)
    stems = sorted(p.with_suffix("") for p in d.glob("*_full_cell.jsonl"))
    for stem in stems:
        if args.only and not any(o in stem.name for o in args.only):
            continue
        analyze_trace(stem)


if __name__ == "__main__":
    main()
