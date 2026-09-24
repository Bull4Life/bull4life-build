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
import os, sys, json, time, urllib.request, urllib.parse, urllib.error, random

TREE_URL  = os.environ.get("TREE_URL", "").rstrip("/")  # improvement_tree.json (public build repo)
SCORE_URL = os.environ.get("SCORE_URL", "").rstrip("/")  # scorecard.json (engine's feedback + hit-rates)
BRANCH    = os.environ.get("BRANCH", "")                # which tree branch this agent owns (by key)
MODEL_NAME= os.environ.get("MODEL_NAME", "?")           # e.g. Qwen3-14B (accountability: who produced this)
MODEL_PARAMS=os.environ.get("MODEL_PARAMS", "?B")       # e.g. 14B  (size -> difficulty band)
BAND      = os.environ.get("BAND", "M")                 # S / M / M+ / L (task-size the model should handle)
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

PERSONA = AGENT.split("-")[1] if AGENT.count("-") >= 1 else AGENT

def load_scorecard():
    """Fetch engine's scorecard -> this agent's own track record + note (so it knows how it's doing)."""
    if not SCORE_URL: return None
    try:
        req = urllib.request.Request(SCORE_URL, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=15) as r: card = json.loads(r.read().decode())
        return card.get("agents", {}).get(PERSONA)
    except Exception as e:
        log("scorecard fetch failed:", e); return None

BR = load_branch()
BRANCH_TITLE = BR["b"]["title"] if BR else LANE
# Identity signature stamped on EVERY post so a bad researcher is traceable to its exact model + size
# (accountability, Admin 2026-09-15). Persona still parses from AGENT which stays at the front of the tag.
SIG = f"{MODEL_NAME}·{MODEL_PARAMS}·band:{BAND}"
OUTPUT_CONTRACT = (BR["tree"].get("output_contract") if BR else
    "Post a [FOR-ENGINE] candidate: proven floor, new hypothesis, an engine-runnable TEST, and the version it could advance.")

# The mission is FORWARD: every finding builds PAST a proven fleet result toward something engine has NOT
# tested. Re-deriving or restating a settled law is useless spam and must be dropped, not posted.
MYSCORE = load_scorecard()
SCORE_LINE = ""
if MYSCORE and MYSCORE.get("posted"):
    SCORE_LINE = (f" Your track record so far: {MYSCORE.get('queued',0)+MYSCORE.get('folded',0)}/"
                  f"{MYSCORE['posted']} posts were useful to engine (hit-rate {MYSCORE.get('hit_rate')}), "
                  f"{MYSCORE.get('rejected',0)} rejected as re-derivation/untestable. Raise that hit-rate: "
                  "more of what engine QUEUES/FOLDS, none of what it REJECTS.")

SYS = (f"You are {AGENT}, a research agent in the Bull4Life trading fleet, on the '{BRANCH_TITLE}' branch. "
       "Your job is NOT to re-prove what the fleet already knows -- the strategy, the bot and the indicator "
       "have been proven many times; re-deriving a settled result is useless spam. Your job is to build PAST "
       "a proven finding toward the NEXT step ENGINE can TEST and fold into a version bump. Be concrete and "
       "honest; if a candidate is weak or just a restatement, SAY SO and drop it. Never fabricate. "
       f"Output contract: {OUTPUT_CONTRACT}{SCORE_LINE}")

# __B4L_LIT_GROUNDING__ (engine 2026-09-24, admin: "send the research fleet where the real info is -> less junk").
# Measured: 150 board posts -> 0 worth reading; the agents invented documents and facts because they had nothing real
# to stand on. Now every study first RETRIEVES real papers (arXiv + OpenAlex, free, no key) for one of the branch's
# curated research_queries, and the model must build on and cite ONLY those - by URL, which the Jev curator re-checks.
LIT_Q_IDX = [0]

def _get(url, timeout=20, _retry=True):
    # arXiv answers 406 to urllib's default 'Accept-Encoding: identity' (measured 2026-09-24); ask for gzip.
    req = urllib.request.Request(url, headers={"User-Agent": "b4l-research/1.0 (engine@bull4life.com)",
                                               "Accept-Encoding": "gzip"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if _retry and e.code in (406, 429, 503):      # both APIs rate-limit bursts; one polite retry
            time.sleep(8)
            return _get(url, timeout, _retry=False)
        raise
    with r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")

def literature(query, n=4):
    """Up to 2n REAL papers for query: [{title, url, year, abstract}]. Fails soft to []."""
    out = []
    try:   # arXiv (Atom)
        import xml.etree.ElementTree as ET
        x = _get("https://export.arxiv.org/api/query?search_query=all:" + urllib.parse.quote(query)
                 + f"&max_results={n}&sortBy=relevance")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(x).findall("a:entry", ns):
            t = " ".join((e.findtext("a:title", "", ns) or "").split())
            u = (e.findtext("a:id", "", ns) or "").strip()
            if t and u:
                out.append({"title": t, "url": u, "year": (e.findtext("a:published", "", ns) or "")[:4],
                            "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split())[:420]})
    except Exception as ex:
        log("arxiv skip:", str(ex)[:80])
    try:   # OpenAlex (the abstract arrives as an inverted index)
        d = json.loads(_get("https://api.openalex.org/works?search=" + urllib.parse.quote(query)
                            + f"&per-page={n}&filter=has_abstract:true&mailto=engine@bull4life.com"))
        for w in d.get("results", []):
            inv = w.get("abstract_inverted_index") or {}
            words = sorted((i, k) for k, v in inv.items() for i in v)
            url = w.get("doi") or w.get("id")
            if w.get("display_name") and url:
                out.append({"title": w["display_name"], "url": url, "year": str(w.get("publication_year") or ""),
                            "abstract": " ".join(k for _, k in words)[:420]})
    except Exception as ex:
        log("openalex skip:", str(ex)[:80])
    return out[: 2 * n]

def grounding():
    """Papers for the next curated research query of this branch (rotates), formatted for the prompt."""
    qs = (BR["b"].get("research_queries") if BR else None) or []
    if not qs:
        return "", []
    q = qs[LIT_Q_IDX[0] % len(qs)]; LIT_Q_IDX[0] += 1
    papers = literature(q)
    if not papers:
        return "", []
    txt = f"REAL PAPERS retrieved just now for '{q}' (the ONLY sources you may use or cite):\n" + "\n".join(
        f"[{i+1}] {p['title']} ({p['year']}) {p['url']}\n    {p['abstract']}" for i, p in enumerate(papers))
    return txt, papers

def study(node, guidance=""):
    """node = a branch seed/frontier dict {proven, frontier} OR a plain question string for advanced threads.
    guidance = recent [ENGINE-FEEDBACK] the agent should steer by (learn what engine values)."""
    if isinstance(node, dict):
        proven = node.get("proven", ""); frontier = node.get("frontier", node.get("q", ""))
        head = (f"PROVEN FLOOR (settled -- do NOT re-derive this):\n{proven}\n\n"
                f"FRONTIER (produce this):\n{frontier}")
    else:
        head = f"Forward question (build past what is proven, do not restate it):\n{node}"
    if guidance:
        head += (f"\n\nENGINE'S RECENT FEEDBACK TO YOU (steer by this -- do more of what engine QUEUED/FOLDED, "
                 f"avoid what it REJECTED):\n{guidance[:600]}")
    lit, papers = grounding()
    if lit:
        head += ("\n\n" + lit + "\n\nBUILD ON THESE PAPERS. Translate one paper's actual method or equation into our "
                 "market (crypto perps, WaveTrend counter ladder, 1/3/6/9/26m bars). Cite it as SOURCE: [n] <url>. "
                 "Never cite anything that is not in this list; never invent a document, a result or a bot rule.")
    draft = ask(SYS, f"{head}\n\nGive your best FORWARD candidate in <=170 words: the new hypothesis/method "
                     "(not the proven floor restated), the mechanism, and one concrete TEST engine can run on "
                     "real data/backtests with an expected result.")
    crit  = ask(SYS, f"Here is a candidate:\n\n{draft}\n\nAdversarially critique it in <=90 words: is it just a "
                     "RESTATEMENT of the proven floor? Is the test actually runnable? Would a pass really advance "
                     "a version? If it fails any of these, say DROP and why.")
    final = ask(SYS, f"{head}\nCandidate: {draft}\nCritique: {crit}\n\nIf the critique said DROP, reply exactly "
                     "'DROP: <one line why>'. Otherwise give the REFINED [FOR-ENGINE] candidate in <=170 words as: "
                     "FLOOR: <proven basis> / HYPOTHESIS: <the new thing> / TEST: <engine-runnable check + expected "
                     "result> / ADVANCES: <which version and how>"
                     + (" / SOURCE: [n] <the exact url from the list>" if lit else "")
                     + ". End with 'CONFIDENCE: low|medium|high'.")
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
    guidance = ""  # latest engine feedback addressed to this agent's persona
    hr = MYSCORE.get("hit_rate") if MYSCORE else None
    cc_post(f"[RESEARCH {AGENT} · {SIG}] online · branch: {BRANCH_TITLE} · FORWARD mission"
            + (f" · hit-rate {hr}" if hr is not None else "") + ".")
    while time.time() < end:
        turn += 1
        msgs, since = cc_read(BOARD, since)
        peer = None
        for m in reversed(msgs):  # newest first
            b = m.get("body", "")
            # engine feedback addressed to us (or the whole board) -> steer the next study
            if "[ENGINE-FEEDBACK" in b and (f"@{PERSONA}" in b or "@all" in b):
                guidance = (b + "\n" + guidance)[:900]
            if peer is None and "[RESEARCH " in b and AGENT not in b:  # newest peer candidate, not our own
                peer = b
        # alternate: verify a peer when we have one, else study
        if peer and turn % 2 == 0:
            try:
                v = verify(peer)
                cc_post(f"[VERIFY {AGENT} · {SIG}] {v}")
                log("posted verify")
            except Exception as e:
                log("verify error:", e)
        else:
            try:
                finding, conf, dropped = study(node, guidance)
                if dropped:
                    # a self-dropped candidate is NOT posted -- that is the anti-spam gate working.
                    log(f"self-DROPPED (no useless post); {finding[:80]}")
                    dropped_streak += 1
                    depth = MAX_DEPTH  # force advance to a new node rather than grind a dead one
                else:
                    label = node.get("id", "q") if isinstance(node, dict) else "q"
                    cc_post(f"[RESEARCH {AGENT} · {SIG}] [{label}] [FOR-ENGINE]\n{finding}")
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
                    cc_post(f"[NEXT {AGENT} · {SIG}] [{lab}] " + (node.get("frontier", "") if isinstance(node, dict) else node)[:220])
            except Exception as e:
                log("study error:", e); time.sleep(15)
        time.sleep(8)
    cc_post(f"[RESEARCH {AGENT} · {SIG}] window done · branch {BRANCH_TITLE} · sleeping until the next spin.")
    log("window complete")

if __name__ == "__main__":
    main()
