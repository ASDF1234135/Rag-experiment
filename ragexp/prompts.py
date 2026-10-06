"""Prompt and Jev question templates (spec section 5). Everything is English."""

QUERY_GEN_SYSTEM = (
    "Write ONE new search query that targets information still missing. "
    "Do not repeat earlier queries. Output only the query."
)
JUDGE_SYSTEM = (
    "Using only the evidence, can the question be fully answered? "
    'Reply as JSON: {"sufficient": true|false}'
)
ANSWER_SYSTEM = (
    "Answer using only the evidence. Cite chunk ids. If the evidence is insufficient, say so."
)
REACT_SYSTEM = (
    "You answer questions by searching a document collection with the `search` tool. "
    "Search as many times as needed, then give the final answer using only retrieved evidence. "
    "Cite chunk ids. If the evidence is insufficient, say so."
)
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the document collection; returns the top matching passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}

JEV_SUFFICIENT = {
    "evidence_sufficient": {
        "type": "noul",
        "instructions": "The passages in `evidence` contain all the information needed to fully answer `question`.",
    }
}
JEV_RANK = {
    "usefulness": {
        "type": "score",
        "instructions": "How useful this passage is for answering the query",
        "criteria": [
            "Unrelated",
            "Related topic, no answer",
            "Contains part of the answer",
            "Directly answers the query",
        ],
    },
    "is_relevant": {"type": "noul", "instructions": "The passage addresses the subject of the query."},
}

FAITHFULNESS_SYSTEM = (
    "You check whether an answer is faithful to the evidence. Break the answer into atomic factual "
    "claims; ignore citations, hedging, and statements that the evidence is insufficient. For each "
    "claim decide whether the evidence directly supports it. Use only the evidence, not outside "
    'knowledge. Reply as JSON: {"claims": [{"claim": "...", "supported": true|false}]}'
)
COMPLETENESS_SYSTEM = (
    "Below are numbered key sentences that are relevant to a question, and an answer to that question. "
    "For each key sentence decide whether the answer conveys the information it contains. "
    'Reply as JSON: {"covered": [<numbers of the key sentences the answer conveys>]}'
)
ANSWER_GRADER_SYSTEM = (
    "You are grading a response against a reference answer. Judge whether the response's final answer "
    "matches the correct answer, allowing for small differences in wording or format but no "
    'substantive differences. Reply as JSON: {"correct": true|false}'
)
