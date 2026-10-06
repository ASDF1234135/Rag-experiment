"""Pilot step 1: one tiny real call per API. Usage: python -m scripts.smoke"""

import json
import sys
import time

from ragexp import prompts
from ragexp.cache import ApiCache
from ragexp.clients import ApiError, EmbedClient, GLMClient, JevClient
from ragexp.config import Config
from ragexp.envfile import load_dotenv
from ragexp.judges import parse_json_object


def step(name, fn):
    t = time.perf_counter()
    try:
        out = fn()
        print(f"[ OK ] {name} ({time.perf_counter() - t:.1f}s) {out}")
        return True
    except ApiError as e:
        print(f"[FAIL] {name}: HTTP {e.status} {str(e)[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
    return False


def main():
    load_dotenv(".env")
    cfg = Config.load("configs/experiment.yaml")
    ok = []
    print(f"glm: {cfg.glm.provider} {cfg.glm.model} @ {cfg.glm.base_url}")
    print(f"judge: {cfg.judge.provider} {cfg.judge.model} @ {cfg.judge.base_url}")

    emb = EmbedClient(cfg.embed)
    ok.append(step("embedding", lambda: (lambda v, m: f"dim={v.shape[1]} tokens={m.input_tokens}")(
        *emb.embed(["smoke test query"], use_cache=False))))

    jev = JevClient(cfg.jev)
    state = {"question": "Who wrote Hamlet?", "evidence": [{"id": "c1", "title": "Hamlet", "text": "Hamlet is a play by William Shakespeare."}]}

    def jev_noul():
        r = jev.ask(state, prompts.JEV_SUFFICIENT)
        return f"noul={r.answers['evidence_sufficient']['noul']} usage=({r.meta.input_tokens},{r.meta.output_tokens})"

    def jev_score():
        r = jev.ask({"query": "author of Hamlet", "passage": state["evidence"][0]}, prompts.JEV_RANK)
        a = r.answers
        return f"score={a['usefulness']['score']:.2f} relevant={a['is_relevant']['noul']:.2f}"

    ok.append(step("jev noul", jev_noul))
    ok.append(step("jev score", jev_score))

    glm = GLMClient(cfg.glm)

    def glm_plain():
        r = glm.chat([{"role": "user", "content": "Reply with the single word: pong"}])
        m = r.meta
        return f"{r.content.strip()[:40]!r} in={m.input_tokens} out={m.output_tokens} reasoning={m.reasoning_tokens}"

    def glm_tool():
        r = glm.chat([{"role": "system", "content": prompts.REACT_SYSTEM},
                      {"role": "user", "content": "Who wrote Hamlet?"}], tools=[prompts.SEARCH_TOOL], tool_choice="auto")
        return f"tool_calls={[(t['name'], t['arguments']) for t in r.tool_calls]} content={r.content[:60]!r} reasoning={r.meta.reasoning_tokens}"

    def glm_json():
        r = glm.chat([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                      {"role": "user", "content": "Question: Who wrote Hamlet?\n\nEvidence:\n[c1] Hamlet\nHamlet is a play by William Shakespeare."}])
        return f"parsed={parse_json_object(r.content)} raw={r.content[:80]!r}"

    ok.append(step("glm plain", glm_plain))
    ok.append(step("glm tool calling", glm_tool))
    ok.append(step("glm judge-style JSON", glm_json))

    judge = GLMClient(cfg.judge)

    def judge_json():
        r = judge.chat([{"role": "system", "content": prompts.FAITHFULNESS_SYSTEM},
                        {"role": "user", "content": "Evidence:\n[c1] Hamlet is a play by William Shakespeare.\n\nAnswer:\nHamlet was written by Shakespeare in 1603."}])
        m = r.meta
        return f"parsed={json.dumps(parse_json_object(r.content))[:150]} in={m.input_tokens} out={m.output_tokens} reasoning={m.reasoning_tokens}"

    ok.append(step("judge faithfulness JSON", judge_json))
    print(f"\n{sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
