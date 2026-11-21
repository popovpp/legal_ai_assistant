# app/graph/prompts.py
ANALYST_SYSTEM = (
    "You are a legal document analyst. Answer strictly from the provided fragments. "
    "Every non-trivial statement MUST be supported by a citation whose `quote` is a "
    "VERBATIM substring of the cited fragment. "
    "Only use chunk_ids that appear in the fragments. Do not invent chunk_ids. "
    "If the fragments do not contain the answer, set `insufficient_evidence=true` and "
    "answer with 'Insufficient evidence in the provided document.'"
)


def build_analyst_prompt(query, mode, chunks, feedback=None):
    context = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | page={c['page']} | score={c['score']:.3f}]\n{c['text']}" for c in chunks
    )
    mode_instructions = {
        "qa": "Answer the question based only on the fragments.",
        "summary": "Summarize the document: key obligations, parties, terms, and dates.",
        "risk": (
            "Identify legal risks, unbalanced clauses, and unfavorable terms. "
            "For each risk provide severity (low/medium/high) and a citation."
        ),
    }
    mode_instr = mode_instructions.get(mode, mode_instructions["qa"])

    fb = ""
    if feedback:
        fb = (
            "\n\nPREVIOUS ATTEMPT FAILED VERIFICATION. "
            f"These chunk_ids had unsupported quotes: {feedback}. "
            "Re-check that each quote is a VERBATIM substring of the cited chunk. "
            "Prefer shorter exact quotes."
        )

    return (
        f"{mode_instr}\n\n"
        f"Question / instruction:\n{query}\n\n"
        f"Document fragments:\n{context}{fb}\n\n"
        "Return JSON with fields: answer, insufficient_evidence, citations, risks."
    )
