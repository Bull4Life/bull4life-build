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

CC_BASE   = os.environ.get("CC_BASE", "").rstrip("/")
CC_TOKEN  = os.environ.get("CC_AGENT_TOKEN", "")
AGENT     = os.environ.get("AGENT", "research-agent")
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
        headers={"Authorization": "Bearer " + CC_TOKEN, "X-CC-Agent": AGENT,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status < 300
    except Exception as e:
        log("cc_post failed:", e); return False

def cc_read(channel, since):
    if not (CC_BASE and CC_TOKEN): return [], since
    url = f"{CC_BASE}/api/messages?channel={urllib.parse.quote(channel)}&since={int(since)}&_cb={int(time.time()*1000)}"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + CC_TOKEN, "X-CC-Agent": AGENT})
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

# ---- the study / verify turns (a mini self-improve: draft -> critique -> refine) ----
SYS = (f"You are {AGENT}, a research agent in the Bull4Life fleet. Lane: {LANE}. "
       "Be concrete and honest. If a claim is weak or unproven, say so. Never fabricate results. "
       "Prefer measured, falsifiable statements over vibes.")

def study(question):
    draft = ask(SYS, f"Study this question and give your best current finding:\n\n{question}\n\n"
                     "Answer in <=150 words: the finding, the mechanism/why, and one concrete way to TEST it.")
    crit  = ask(SYS, f"Here is a draft finding:\n\n{draft}\n\nAdversarially critique it in <=80 words: "
                     "what's the weakest claim, and is it fitted/overclaimed?")
    final = ask(SYS, f"Question: {question}\nDraft: {draft}\nCritique: {crit}\n\n"
                     "Give the REFINED finding in <=150 words, keeping only what survives the critique. "
                     "End with 'CONFIDENCE: low|medium|high' and 'TEST: <one concrete check>'.")
    conf = "low"
    for c in ("high", "medium", "low"):
        if f"CONFIDENCE: {c}" in final.lower() or f"confidence: {c}" in final.lower(): conf = c; break
    return final, conf

def verify(peer_text):
    return ask(SYS, f"A peer agent posted this finding:\n\n{peer_text[:1500]}\n\n"
                    "Adversarially verify it in <=100 words. Try to REFUTE it. Verdict: HOLDS or REFUTED, and why. "
                    "Default to REFUTED if the evidence is thin or the breadth is a coin-flip.")

SEED_QUESTIONS = [
    f"What is the single highest-leverage open question in '{LANE}' right now, and a first answer?",
]

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
    thread = SEED_QUESTIONS[0]; depth = 0; turn = 0
    cc_post(f"[RESEARCH {AGENT}] online · lane: {LANE} · joining the swarm board.")
    while time.time() < end:
        turn += 1
        msgs, since = cc_read(BOARD, since)
        peer = None
        for m in reversed(msgs):
            b = m.get("body", "")
            if "[RESEARCH " in b and AGENT not in b:  # a peer's finding, not our own
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
                finding, conf = study(thread)
                cc_post(f"[RESEARCH {AGENT}] Q: {thread[:120]}\n{finding}")
                log(f"posted study (conf={conf}, depth={depth})")
                depth += 1
                if conf == "high" or depth >= MAX_DEPTH:
                    # ADVANCE (proven or capped) -> next question from a peer thread or a fresh angle
                    nxt = ask(SYS, f"Given the finished thread '{thread}', propose ONE sharper NEXT question "
                                   f"in lane '{LANE}'. Reply with only the question.")
                    thread = nxt.strip().split("\n")[0][:200] or SEED_QUESTIONS[0]; depth = 0
                    cc_post(f"[NEXT {AGENT}] {thread}")
            except Exception as e:
                log("study error:", e); time.sleep(15)
        time.sleep(8)
    cc_post(f"[RESEARCH {AGENT}] window done · sleeping until the next spin.")
    log("window complete")

if __name__ == "__main__":
    main()
