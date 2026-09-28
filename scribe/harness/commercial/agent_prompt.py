"""Builds the task prompt that every commercial agent receives from the task's TaskSpec."""

from __future__ import annotations


def prompt(spec: dict) -> str:
    o = spec["output_spec"]
    return (
        f"Write a scientific literature review on the topic: \"{spec['question']}\".\n\n"
        f"Audience: {spec['audience']}.\n"
        f"Target length: about {o['target_words']} words of body text.\n\n"
        "Requirements:\n"
        "- Research the literature using the provided search and fetch tools; they are your only source of papers.\n"
        "- Organise the review into titled sections (markdown headings).\n"
        "- Support statements with numeric inline citations like [1], [2].\n"
        "- End with a 'References' section listing every cited paper as: [n] Title. PMID: <pmid>. <url>\n"
    )
