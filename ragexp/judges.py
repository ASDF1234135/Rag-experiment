from __future__ import annotations

import json
import re

from .clients import CallMeta, GLMClient, JevClient
from . import prompts


def _evidence_user(question: str, evidence_text: str, extra: str = "") -> str:
    return f"Question: {question}\n\n{extra}Evidence:\n{evidence_text}"


def parse_json_flag(text: str, key: str) -> bool | None:
    """Pull {"<key>": true|false} out of a model reply (tolerates code fences / chatter)."""
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get(key)
            if isinstance(v, bool):
                return v
        except (json.JSONDecodeError, AttributeError):
            pass
    m = re.search(rf'"{key}"\s*:\s*(true|false)', text, flags=re.I)
    return (m.group(1).lower() == "true") if m else None


class LLMJudge:
    """G3: binary sufficiency decision from the same input the Jev judge sees."""

    def __init__(self, llm: GLMClient):
        self.llm = llm

    def judge(self, question: str, evidence_text: str, salt: str = "") -> tuple[bool, CallMeta, bool]:
        r = self.llm.chat(
            [{"role": "system", "content": prompts.JUDGE_SYSTEM},
             {"role": "user", "content": _evidence_user(question, evidence_text)}],
            salt=salt,
        )
        flag = parse_json_flag(r.content, "sufficient")
        return bool(flag), r.meta, flag is not None  # unparsable -> "insufficient", flagged


class JevJudge:
    """G4/G5: P(evidence is sufficient) from Jev's `noul`."""

    def __init__(self, jev: JevClient):
        self.jev = jev

    def judge(self, question: str, evidence_items: list[dict]) -> tuple[float, CallMeta]:
        r = self.jev.ask({"question": question, "evidence": evidence_items}, prompts.JEV_SUFFICIENT)
        return float(r.answers["evidence_sufficient"]["noul"]), r.meta


class QueryGenerator:
    def __init__(self, llm: GLMClient):
        self.llm = llm

    def generate(self, question: str, evidence_text: str, prior_queries: list[str],
                 salt: str = "") -> tuple[str, CallMeta]:
        earlier = "Earlier queries:\n" + "\n".join(f"- {q}" for q in prior_queries) + "\n\n"
        r = self.llm.chat(
            [{"role": "system", "content": prompts.QUERY_GEN_SYSTEM},
             {"role": "user", "content": _evidence_user(question, evidence_text, earlier)}],
            salt=salt,
        )
        lines = [ln.strip().strip('"').strip() for ln in r.content.strip().splitlines() if ln.strip()]
        return (lines[0] if lines else question), r.meta


def generate_answer(llm: GLMClient, question: str, evidence_text: str, salt: str = "") -> tuple[str, CallMeta]:
    r = llm.chat(
        [{"role": "system", "content": prompts.ANSWER_SYSTEM},
         {"role": "user", "content": _evidence_user(question, evidence_text)}],
        salt=salt,
    )
    return r.content, r.meta


def grade_answer(judge_llm: GLMClient, question: str, reference: str, response: str) -> tuple[bool, CallMeta]:
    """Answer correctness against a gold answer. Judge model is TBD in the spec; one fixed model for all groups."""
    r = judge_llm.chat(
        [{"role": "system", "content": prompts.ANSWER_GRADER_SYSTEM},
         {"role": "user", "content": f"[question]: {question}\n[correct_answer]: {reference}\n[response]: {response}"}]
    )
    return bool(parse_json_flag(r.content, "correct")), r.meta


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a model reply (tolerates code fences and chatter; handles nesting)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def judge_faithfulness(judge_llm: GLMClient, evidence_text: str, answer: str) -> dict:
    """Claim-level faithfulness against the evidence the group actually used (spec section 7).

    Returns {"faithfulness": supported/claims or None, "fully_faithful": bool or None, "n_claims": int,
    "parse_ok": bool, "meta": CallMeta}. An answer with no factual claims (e.g. "the evidence is
    insufficient") has faithfulness None: it is excluded from the mean and counted as an abstention."""
    r = judge_llm.chat([
        {"role": "system", "content": prompts.FAITHFULNESS_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{evidence_text}\n\nAnswer:\n{answer}"},
    ])
    obj = parse_json_object(r.content)
    claims = obj.get("claims") if obj else None
    if not isinstance(claims, list):
        return {"faithfulness": None, "fully_faithful": None, "n_claims": 0, "parse_ok": False, "meta": r.meta}
    flags = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
    if not flags:
        return {"faithfulness": None, "fully_faithful": None, "n_claims": 0, "parse_ok": True, "meta": r.meta}
    return {"faithfulness": sum(flags) / len(flags), "fully_faithful": all(flags), "n_claims": len(flags),
            "parse_ok": True, "meta": r.meta}


def judge_completeness(judge_llm: GLMClient, question: str, answer: str, key_sentences: list[str]) -> dict:
    """Share of the annotated relevant sentences whose information the answer conveys.
    Returns {"completeness": float or None, "n_key": int, "parse_ok": bool, "meta": CallMeta}."""
    if not key_sentences:
        return {"completeness": None, "n_key": 0, "parse_ok": True, "meta": None}
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(key_sentences, 1))
    r = judge_llm.chat([
        {"role": "system", "content": prompts.COMPLETENESS_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nKey sentences:\n{numbered}\n\nAnswer:\n{answer}"},
    ])
    obj = parse_json_object(r.content)
    cov = obj.get("covered") if obj else None
    if not isinstance(cov, list):
        return {"completeness": None, "n_key": len(key_sentences), "parse_ok": False, "meta": r.meta}
    ids = set()
    for x in cov:
        try:
            ids.add(int(float(x)))
        except (TypeError, ValueError):
            pass
    return {"completeness": len(ids & set(range(1, len(key_sentences) + 1))) / len(key_sentences),
            "n_key": len(key_sentences), "parse_ok": True, "meta": r.meta}
