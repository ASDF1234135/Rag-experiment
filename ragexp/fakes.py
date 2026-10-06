"""Deterministic offline stand-ins for the embedding model, GLM and Jev, used by tests and `ragexp demo`.

They mimic the *interfaces and usage accounting* of the real clients, not their quality. The fake
corpus is two-hop: the question names ENTITY_k; the doc about ENTITY_k mentions BRIDGE_k; the doc
about BRIDGE_k contains ANSWER_k. A good multi-round flow must find both documents.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .clients import CallMeta, JevResult, LLMResult
from .index import normalize
from .types import Doc, Question

WORD = re.compile(r"[A-Za-z_0-9]+")


def _words(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


class FakeEmbedder:
    DIM = 512

    def embed(self, texts: list[str], use_cache: bool = True, est_tokens=None):
        out = np.zeros((len(texts), self.DIM), np.float32)
        for i, t in enumerate(texts):
            for w in _words(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = sum(len(_words(t)) for t in texts)
        return normalize(out), CallMeta(provider="fake", model="fake-embed", input_tokens=n)


def _meta(inp: str, out: str) -> CallMeta:
    return CallMeta(provider="fake", model="fake", input_tokens=len(_words(inp)), output_tokens=len(_words(out)),
                    latency_ms=1.0)


class FakeLLM:
    """Routes on the system prompt, like the real prompts in prompts.py."""

    def __init__(self):
        self.calls = 0
        self.salts: list[str] = []

    def chat(self, messages, tools=None, tool_choice=None, salt: str = "") -> LLMResult:
        self.calls += 1
        self.salts.append(salt)
        system, user = messages[0]["content"], messages[-1]["content"]
        allin = " ".join(str(m.get("content") or "") for m in messages)
        if system.startswith("Write ONE new search query"):
            question = user.split("\n")[0]
            bridges = [b for b in re.findall(r"BRIDGE_\d+", user.split("Evidence:")[-1]) if b not in question]
            out = f"{bridges[0]} details" if bridges else "more information"
            return LLMResult(out, meta=_meta(user, out))
        if system.startswith("Using only the evidence"):
            ok = "ANSWER_" in user.split("Evidence:")[-1]
            out = json.dumps({"sufficient": ok})
            return LLMResult(out, meta=_meta(user, out))
        if system.startswith("Answer using only"):
            m = re.search(r"ANSWER_\d+", user.split("Evidence:")[-1])
            out = f"The answer is {m.group(0)}." if m else "The evidence is insufficient."
            return LLMResult(out, meta=_meta(user, out))
        if system.startswith("You answer questions by searching"):  # ReAct
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            seen = " ".join(m["content"] for m in tool_msgs)
            m = re.search(r"ANSWER_\d+", seen)
            if m or tool_choice == "none":
                out = f"The answer is {m.group(0)}." if m else "The evidence is insufficient."
                return LLMResult(out, meta=_meta(allin, out))
            br = re.findall(r"BRIDGE_\d+", seen)
            query = f"{br[0]} details" if br else messages[1]["content"]
            tc = [{"id": f"call{len(tool_msgs)}", "name": "search", "arguments": json.dumps({"query": query})}]
            return LLMResult("", tool_calls=tc, meta=_meta(allin, query))
        if system.startswith("You check whether an answer is faithful"):
            evidence, answer = user.split("Evidence:\n", 1)[1].split("\n\nAnswer:\n", 1)
            m = re.search(r"ANSWER_\d+", answer)
            claims = [] if not m else [{"claim": answer.strip(), "supported": m.group(0) in evidence}]
            out = json.dumps({"claims": claims})
            return LLMResult(out, meta=_meta(user, out))
        if system.startswith("Below are numbered key sentences"):
            sents = re.findall(r"^(\d+)\. (.*)$", user, flags=re.M)
            answer = user.split("Answer:\n", 1)[1]
            covered = [int(i) for i, t in sents if any(w in answer for w in re.findall(r"(?:ENTITY|BRIDGE|ANSWER)_\d+", t))]
            out = json.dumps({"covered": covered})
            return LLMResult(out, meta=_meta(user, out))
        if system.startswith("You are grading"):
            ok = bool(re.search(r"\[correct_answer\]: (\S+)", user) and
                      re.search(r"\[correct_answer\]: (\S+)", user).group(1) in user.split("[response]:")[-1])
            out = json.dumps({"correct": ok})
            return LLMResult(out, meta=_meta(user, out))
        raise AssertionError(f"unexpected system prompt: {system[:40]}")


class FakeJev:
    """noul: 0.9 when the evidence contains an ANSWER_ token, else 0.2. score: word overlap level."""

    class cfg:
        model = "fake-jev"

    def ask(self, state, questions) -> JevResult:
        answers = {}
        for name, qd in questions.items():
            if name == "evidence_sufficient":
                ev = " ".join(e["text"] for e in state["evidence"])
                answers[name] = {"type": "noul", "noul": 0.9 if "ANSWER_" in ev else 0.2}
            elif name == "usefulness":
                qw, pw = set(_words(state["query"])), set(_words(state["passage"]["text"] + " " + state["passage"]["title"]))
                ov = len(qw & pw) / max(1, len(qw))
                answers[name] = {"type": "score", "score": 3.0 * ov}
            elif name == "is_relevant":
                qw, pw = set(_words(state["query"])), set(_words(state["passage"]["text"]))
                answers[name] = {"type": "noul", "noul": min(1.0, len(qw & pw) / max(1, len(qw)))}
        text = json.dumps(state)
        return JevResult(answers, CallMeta(provider="typesafe", model="fake-jev",
                                           input_tokens=len(_words(text)), output_tokens=3, latency_ms=1.0))


def make_two_hop_corpus(n_questions: int = 6, n_distractors: int = 60) -> tuple[list[Doc], list[Question]]:
    docs: list[Doc] = []
    qs: list[Question] = []
    filler = "lorem ipsum dolor sit amet consectetur " * 6
    for k in range(n_questions):
        a, b = f"doc_a{k}", f"doc_b{k}"
        docs.append(Doc(a, f"About ENTITY_{k}", f"ENTITY_{k} is linked to BRIDGE_{k}. {filler}"))
        docs.append(Doc(b, f"About BRIDGE_{k}", f"BRIDGE_{k} details: holds ANSWER_{k}. {filler}"))
        qs.append(Question(f"q{k}", f"Which result relates to ENTITY_{k}?", f"ANSWER_{k}",
                           gold_doc_ids=[a, b], evidence_doc_ids=[a, b],
                           meta={"relevant_sentences": [f"ENTITY_{k} is linked to BRIDGE_{k}.",
                                                        f"BRIDGE_{k} details: holds ANSWER_{k}."]}))
    for i in range(n_distractors):
        docs.append(Doc(f"dist{i}", f"Distractor {i}", f"Unrelated topic number {i} about gardening and weather. {filler}"))
    return docs, qs
