#!/usr/bin/env python3
"""research_agent.py - one independent research agent for a GitHub runner slot.

Runs on a 4-core/16GB ubuntu-latest runner beside a local llama-server (a 7-14B GGUF).
It is NOT pooled - each runner is a standalone agent. The agents "improve each other" through
the Command Center: each posts findings to a shared research channel and periodically VERIFIES a
peer's finding (adversarial), so the swarm cross-checks instead of grading its own homework.

Loop (until the window runs out, leaving margin before the 6h GitHub cap):
  1. read the research board (CC channel) since our cursor -> recent peer findings
  2. alternate STUDY (advance a question on our lane) and VERIFY (adversarially check a peer)
  3. post the finding/verdict to the CC, tagged so engine can materialize proven ones to golden
  4. quality gate: DEEPEN / ADVANCE / STUCK-THIN - park a stuck thread, pull a fresh question
     (busy != improving; never loop past MAX_DEPTH on one thread)

Env (from the workflow):
  CC_BASE, CC_AGENT_TOKEN   - Command Center endpoint + this agent's token (repo secret)
  AGENT                     - this agent's name/persona, e.g. research-logic-1
  LANE                      - the study lane/persona prompt seed, e.g. "formal logic & proof"
  MODEL_URL                 - local llama-server, default http://127.0.0.1:8080
  BOARD                     - CC channel for the swarm, default "group"
  WINDOW_SEC                - seconds to run, default 19800 (5.5h)
  MAX_DEPTH                 - iterations on one thread before ADVANCE, default 4
"""
import os, sys, json, time, urllib.request, urllib.parse, random

TREE_URL  = os.environ.get("TREE_URL", "").rstrip("/")  # improvement_tree.json (public build repo)
BRANCH    = os.environ.get("BRANCH", "")                # which tree branch this agent owns (by key)
CC_BASE   = os.environ.get("CC_BASE", "").rstrip("/")
CC_TOKEN  = os.environ.get("CC_AGENT_TOKEN", "")
AGENT     = os.environ.get("AGENT", "research-agent")
# CC binds a token to ONE agent identity: a mismatched X-CC-Agent header is rejected 401. The swarm
# shares engine's token, so it POSTS under the authorized identity (CC_POST_AS, e.g. "engine") while
# keeping its own persona (AGENT) in the [RESEARCH <persona>] body tag that engine reads. Default to
# AGENT so a runner given its OWN rostered token still posts under its own name.
CC_POST_AS = os.environ.get("CC_POST_AS", AGENT)
LANE      = os.environ.get("LANE", "general reasoning and research")
MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:8080").rstrip("/")
BOARD     = os.environ.get("BOARD", "group")
WINDOW    = int(os.environ.get("WINDOW_SEC", "19800"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "4"))
MARGIN    = 300  # stop 5 min before the window so the post + clean exit land

def log(*a): print(f"[{AGENT}]", *a, flush=True)

# ---- Command Center (same contract as tools/cc_say.py / cc_read.py) ----
def cc_post(body, channel=None):
    if not (CC_BASE and CC_TOKEN):
        log("no CC creds - printing instead:\n", body[:400]); return False
    data = json.dumps({"channel": channel or BOARD, "body": body[:3500]}).encode()
    req = urllib.request.Request(CC_BASE + "/api/message", data=data, method="POST",
        headers={"Authorization": "Bearer " + CC_TOKEN, "X-CC-Agent": CC_POST_AS,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status < 300
    except Exception as e:
        log("cc_post failed:", e); return False

def cc_read(channel, since):
    if not (CC_BASE and CC_TOKEN): return [], since
    url = f"{CC_BASE}/api/messages?channel={urllib.parse.quote(channel)}&since={int(since)}&_cb={int(time.time()*1000)}"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + CC_TOKEN, "X-CC-Agent": CC_POST_AS})
    try:
        with urllib.request.urlopen(req, timeout=15) as r: d = json.loads(r.read().decode())
        msgs = d.get("messages") or d.get("items") or (d if isinstance(d, list) else [])
        hi = since
        for m in msgs: hi = max(hi, int(m.get("seq", 0)))
        return msgs, hi
    except Exception as e:
        log("cc_read failed:", e); return [], since

# ---- local model (llama-server OpenAI-compatible endpoint) ----
def ask(system, user, max_tokens=700, temperature=0.5):
    body = json.dumps({"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": max_tokens, "temperature": temperature,
                       "stream": False}).encode()
    req = urllib.request.Request(MODEL_URL + "/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"]["content"] or "").strip()

# ---- the improvement tree (the forward mission: build PAST proven, never re-derive it) ----
def load_branch():
    """Fetch improvement_tree.json and return this agent's branch (by BRANCH key, else persona match)."""
    if not TREE_URL:
        return None
    try:
        req = urllib.request.Request(TREE_URL, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=20) as r:
            tree = json.loads(r.read().decode())
    except Exception as e:
        log("tree fetch failed:", e); return None
    persona = AGENT.split("-")[1] if AGENT.count("-") >= 1 else ""
    for b in tree.get("branches", []):
        if BRANCH and b.get("key") == BRANCH: return {"tree": tree, "b": b}
        if not BRANCH and b.get("persona") == persona: return {"tree": tree, "b": b}
    # fall back to the first branch so the agent still runs on-mission
    bl = tree.get("branches", [])
    return {"tree": tree, "b": bl[0]} if bl else None

BR = load_branch()
BRANCH_TITLE = BR["b"]["title"] if BR else LANE
OUTPUT_CONTRACT = (BR["tree"].get("output_contract") if BR else
    "Post a [FOR-ENGINE] candidate: proven floor, new hypothesis, an engine-runnable TEST, and the version it could advance.")

# The mission is FORWARD: every finding builds PAST a proven fleet result toward something engine has NOT
# tested. Re-deriving or restating a settled law is useless spam and must be dropped, not posted.
SYS = (f"You are {AGENT}, a research agent in the Bull4Life trading fleet, on the '{BRANCH_TITLE}' branch. "
       "Your job is NOT to re-prove what the fleet already knows -- the strategy, the bot and the indicator "
       "have been proven many times; re-deriving a settled result is useless spam. Your job is to build PAST "
       "a proven finding toward the NEXT step ENGINE can TEST and fold into a version bump. Be concrete and "
       "honest; if a candidate is weak or just a restatement, SAY SO and drop it. Never fabricate. "
       f"Output contract: {OUTPUT_CONTRACT}")

def study(node):
    """node = a branch seed/frontier dict {proven, frontier} OR a plain question string for advanced threads."""
    if isinstance(node, dict):
        proven = node.get("proven", ""); frontier = node.get("frontier", node.get("q", ""))
        head = (f"PROVEN FLOOR (settled -- do NOT re-derive this):\n{proven}\n\n"
                f"FRONTIER (produce this):\n{frontier}")
    else:
        head = f"Forward question (build past what is proven, do not restate it):\n{node}"
    draft = ask(SYS, f"{head}\n\nGive your best FORWARD candidate in <=170 words: the new hypothesis/method "
                     "(not the proven floor restated), the mechanism, and one concrete TEST engine can run on "
                     "real data/backtests with an expected result.")
    crit  = ask(SYS, f"Here is a candidate:\n\n{draft}\n\nAdversarially critique it in <=90 words: is it just a "
                     "RESTATEMENT of the proven floor? Is the test actually runnable? Would a pass really advance "
                     "a version? If it fails any of these, say DROP and why.")
    final = ask(SYS, f"{head}\nCandidate: {draft}\nCritique: {crit}\n\nIf the critique said DROP, reply exactly "
                     "'DROP: <one line why>'. Otherwise give the REFINED [FOR-ENGINE] candidate in <=170 words as: "
                     "FLOOR: <proven basis> / HYPOTHESIS: <the new thing> / TEST: <engine-runnable check + expected "
                     "result> / ADVANCES: <which version and how>. End with 'CONFIDENCE: low|medium|high'.")
    conf = "low"
    for c in ("high", "medium", "low"):
        if f"confidence: {c}" in final.lower(): conf = c; break
    dropped = final.strip().upper().startswith("DROP")
    return final, conf, dropped

def verify(peer_text):
    return ask(SYS, f"A peer posted this [FOR-ENGINE] candidate:\n\n{peer_text[:1500]}\n\n"
                    "Adversarially verify it in <=110 words. FIRST decide: is it FORWARD (a new testable step) or "
                    "just a RESTATEMENT of something already proven? A restatement is REFUTED automatically. "
                    "Then: is the TEST engine-runnable and would a pass advance a version? Verdict: HOLDS or REFUTED, "
                    "and why. Default REFUTED when thin, vague, or not actually new.")

# Seed the loop from THIS branch's forward nodes (proven floor -> frontier), not a generic lane question.
SEED_NODES = (BR["b"]["seed_nodes"] if BR else
              [f"What is one FORWARD, engine-testable step in '{LANE}' that builds past a proven result?"])

def main():
    log(f"start · lane='{LANE}' · model={MODEL_URL} · window={WINDOW}s · board={BOARD}")
    # wait for the local model to be ready
    for _ in range(60):
        try:
            ask("ping", "reply OK", max_tokens=3, temperature=0); break
        except Exception:
            time.sleep(10)
    end = time.time() + WINDOW - MARGIN
    since = 0
    node_i = 0
    node = SEED_NODES[0]; depth = 0; turn = 0
    dropped_streak = 0
    cc_post(f"[RESEARCH {AGENT}] online · branch: {BRANCH_TITLE} · FORWARD mission (build past proven, feed engine).")
    while time.time() < end:
        turn += 1
        msgs, since = cc_read(BOARD, since)
        peer = None
        for m in reversed(msgs):
            b = m.get("body", "")
            if "[RESEARCH " in b and AGENT not in b:  # a peer's candidate, not our own
                peer = b; break
        # alternate: verify a peer when we have one, else study
        if peer and turn % 2 == 0:
            try:
                v = verify(peer)
                cc_post(f"[VERIFY {AGENT}] {v}")
                log("posted verify")
            except Exception as e:
                log("verify error:", e)
        else:
            try:
                finding, conf, dropped = study(node)
                if dropped:
                    # a self-dropped candidate is NOT posted -- that is the anti-spam gate working.
                    log(f"self-DROPPED (no useless post); {finding[:80]}")
                    dropped_streak += 1
                    depth = MAX_DEPTH  # force advance to a new node rather than grind a dead one
                else:
                    label = node.get("id", "q") if isinstance(node, dict) else "q"
                    cc_post(f"[RESEARCH {AGENT}] [{label}] [FOR-ENGINE]\n{finding}")
                    log(f"posted candidate (conf={conf}, depth={depth})")
                    dropped_streak = 0
                depth += 1
                if conf == "high" or depth >= MAX_DEPTH:
                    # ADVANCE: next branch node, or (nodes exhausted) ask for a sharper FORWARD child
                    node_i += 1
                    if node_i < len(SEED_NODES):
                        node = SEED_NODES[node_i]
                    else:
                        floor = node.get("frontier","") if isinstance(node, dict) else str(node)
                        nxt = ask(SYS, f"We just worked: {floor[:200]}\nPropose ONE sharper FORWARD child question "
                                       f"on the '{BRANCH_TITLE}' branch that engine has NOT tested. Only the question.")
                        node = nxt.strip().split("\n")[0][:220]
                    depth = 0
                    lab = node.get("id","next") if isinstance(node, dict) else "next"
                    cc_post(f"[NEXT {AGENT}] [{lab}] " + (node.get("frontier", "") if isinstance(node, dict) else node)[:220])
            except Exception as e:
                log("study error:", e); time.sleep(15)
        time.sleep(8)
    cc_post(f"[RESEARCH {AGENT}] window done · branch {BRANCH_TITLE} · sleeping until the next spin.")
    log("window complete")

if __name__ == "__main__":
    main()
