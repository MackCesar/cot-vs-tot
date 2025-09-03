SYSTEM_REASONING = (
    "You are a careful reasoning assistant. Think step-by-step, keep calculations explicit, "
    "and return the final answer clearly on the last line after 'FINAL:'."
)

def cot_prompt(question: str) -> str:
    return f"{question}\n\nPlease reason step by step. Then give the final answer after the word 'FINAL:'."

def sc_cot_prompt(question: str) -> str:
    return cot_prompt(question)

def tot_root_prompt(question: str) -> str:
    return (f"Problem: {question}\n"
            "You will plan a solution as a search tree of short steps. "
            "Propose 2-3 alternative next steps. Keep each step under 2 sentences.")

def tot_refine_prompt(question: str, partial: str) -> str:
    return (f"Problem: {question}\n"
            f"Current partial reasoning:\n{partial}\n\n"
            "Propose 2-3 alternative next steps to extend or correct the reasoning.")

def voter_prompt(question: str, candidates: list[str]) -> str:
    joined = "\n---\n".join(candidates)
    return (f"Consider the problem:\n{question}\n\n"
            f"Here are candidate solutions:\n{joined}\n\n"
            "Choose the best one and write ONLY a short justification followed by 'FINAL:' with the final answer.")