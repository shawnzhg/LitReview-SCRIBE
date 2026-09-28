"""Per-agent configuration of view_builder.py: the system keys each external agent adds to the
boards and their roster declarations."""

from __future__ import annotations

_REF_NOTE = "fixed input: the task's reference list. "
_POOL_NOTE = "same pool: the frozen pool. "
_MCP_IF = "mcp(pool_service_bm25 plain_search top-20 + by-id fetch)"

_OPENAI = ("OpenAI Responses API server-side tool loop, gpt-5.6-luna, reasoning effort none; one tool: an MCP server over the "
           "pool service (search and by-id fetch of title, abstract and year; the evaluated review suppressed).")
_CLAUDE_CODE = ("Claude Code CLI headless, claude-sonnet-5, effort low, its own agent loop and system prompt; no built-in tools; "
                "the only tools are the pool MCP server of the OpenAI rows; same prompt.")
_CLAUDE_SCIENCE = ("Claude Science workbench with its own agent loop, literature-review skill and reviewer, claude-sonnet-5, "
                   "effort low; the only literature source is the pool MCP server; same prompt plus the connector name.")
_GEMINI = ("Gemini Deep Research in the web app, one new chat per run, plan accepted unedited, extended thinking off; citation "
           "marks mapped through the list they index.")
_ELICIT = ("Elicit Systematic Review API with Elicit's own models: gather = PubMed '<pmid>[pmid]' searches over the task's "
           "reference list; no screening; generated extraction and report; Elicit reads full text where it finds it.")

_GEMINI_COMMON = dict(vendor_model="gemini-3.8-flash", model_group="gemini-3.8-flash", backbone="gemini-3.8-flash",
                      temperature="Gemini app default (not settable)", retrieval_budget="Gemini Deep Research (not metered)",
                      snapshot_note="Gemini Deep Research", label="Gemini Deep Research (3.8 Flash)")

AGENTS = {
    "openai_tool_loop": dict(
        keys={
            "openai_luna_mcp.ref": dict(dataset="fixed_input", template="gpt_ref", vendor_model="gpt-5.6-luna",
                                        label="OpenAI tool loop, gpt-5.6-luna + pool MCP (reference list)",
                                        interface=_MCP_IF + ", reference list",
                                        temperature="1.0 (provider default, none sent; reasoning none)",
                                        notes=_REF_NOTE + _OPENAI),
            "openai_luna_mcp": dict(dataset="same_pool", template="gpt_self", vendor_model="gpt-5.6-luna",
                                    label="OpenAI tool loop, gpt-5.6-luna + pool MCP",
                                    interface=_MCP_IF + ", full pool",
                                    temperature="1.0 (provider default, none sent; reasoning none)",
                                    notes=_POOL_NOTE + _OPENAI),
        }),
    "claude_code": dict(
        keys={
            "claude_sonnet_mcp.ref": dict(dataset="fixed_input", template="gpt_ref", vendor_model="claude-sonnet-5",
                                          model_group="claude_sonnet5", backbone="claude-sonnet-5",
                                          label="Claude Code (Sonnet 5, effort low) + pool MCP (reference list)",
                                          interface=_MCP_IF + ", reference list",
                                          temperature="Claude Code default (not set; effort low)",
                                          notes=_REF_NOTE + _CLAUDE_CODE),
            "claude_sonnet_mcp": dict(dataset="same_pool", template="gpt_self", vendor_model="claude-sonnet-5",
                                      model_group="claude_sonnet5", backbone="claude-sonnet-5",
                                      label="Claude Code (Sonnet 5, effort low) + pool MCP",
                                      interface=_MCP_IF + ", full pool",
                                      temperature="Claude Code default (not set; effort low)",
                                      notes=_POOL_NOTE + _CLAUDE_CODE),
        }),
    "claude_science": dict(
        keys={
            "claude_science_mcp.ref": dict(dataset="fixed_input", template="gpt_ref", vendor_model="claude-sonnet-5",
                                           model_group="claude_sonnet5", backbone="claude-sonnet-5",
                                           label="Claude Science (Sonnet 5, effort low) + pool MCP (reference list)",
                                           interface=_MCP_IF + ", reference list",
                                           temperature="Claude Science default (not set; effort low)",
                                           notes=_REF_NOTE + _CLAUDE_SCIENCE),
            "claude_science_mcp": dict(dataset="same_pool", template="gpt_self", vendor_model="claude-sonnet-5",
                                       model_group="claude_sonnet5", backbone="claude-sonnet-5",
                                       label="Claude Science (Sonnet 5, effort low) + pool MCP",
                                       interface=_MCP_IF + ", full pool",
                                       temperature="Claude Science default (not set; effort low)",
                                       notes=_POOL_NOTE + _CLAUDE_SCIENCE),
        }),
    "gemini_deep_research": dict(
        keys={
            "gemini_web_dr.ref": dict(_GEMINI_COMMON, dataset="fixed_input", template="gpt_ref",
                                      interface="the task's reference bundle as an uploaded file",
                                      notes=_REF_NOTE + _GEMINI),
            "gemini_web_dr": dict(_GEMINI_COMMON, dataset="same_pool", template="gpt_self",
                                  interface="Gemini Deep Research search, PubMed and PMC under the task's cutoff",
                                  notes=_POOL_NOTE + _GEMINI),
        }),
    "elicit": dict(
        keys={
            "elicit_sr.ref": dict(dataset="fixed_input", template="gpt_ref", vendor_model="elicit (own models, not selectable)",
                                  model_group="elicit", backbone="proprietary (Elicit's own models, not selectable)",
                                  label="Elicit Systematic Review (fixed input only)",
                                  interface="Elicit SR API: gather = PubMed '<pmid>[pmid]' searches over the reference list; "
                                            "no screening; server-side extraction and report",
                                  temperature="n/a (Elicit server-side; not settable)",
                                  notes=_REF_NOTE + _ELICIT,
                                  level_gate={"record": {"level": "n/a (external agent; no run level)"},
                                              "retrieval_budget": "none (entry = the task's reference list)",
                                              "units": "report", "snapshot": "Elicit's PubMed index",
                                              "allow_partial": False}),
        }),
}

EXT_KEYS = {k: dict(v, agent=a) for a, cfg in AGENTS.items() for k, v in cfg["keys"].items()}
