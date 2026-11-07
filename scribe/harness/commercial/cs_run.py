"""Browser steps of one Claude Science run: new project, model and effort set, prompt sent, sandboxed
code-execution cards allowed and every other card denied, and page saved."""

import json, os, re, time
from pathlib import Path

OUT = Path(os.environ["COMMERCIAL_OUT"])
PROMPTS = Path(os.environ["CS_PROMPTS"])
ALLOW_TITLES = ("Run a shell command", "Run Python", "Run code", "Run R", "Run a command", "Execute")


def safe(s):
    return re.sub(r"[0-9a-f]{24,}", "<HEX>", s)


def handle_cards(page, log):
    deny = page.get_by_role("button", name="Deny", exact=True)
    if not deny.count():
        return False
    titles = page.get_by_text(re.compile(r"^[A-Z][^\n]{3,80}\?$")).all()
    title = titles[-1].inner_text().strip() if titles else ""
    code = ""
    try:
        code = page.locator("pre, code").last.inner_text()[:400]
    except Exception:
        pass
    ok = any(title.startswith(t) for t in ALLOW_TITLES)
    allow = page.get_by_role("button", name=re.compile(r"^Allow"))
    (allow.first if ok and allow.count() else deny.first).click()
    log({"card": safe(title), "code": safe(code), "decision": "allow" if ok else "deny"})
    page.wait_for_timeout(2000)
    return True


def set_model_effort(page, model="Sonnet 5", effort="Low"):
    mb = page.get_by_role("button", name=re.compile(r"^Model: "))
    mb.last.click(); page.wait_for_timeout(1200)
    page.get_by_role("option", name=re.compile("^" + re.escape(model))).or_(
        page.get_by_role("menuitemradio", name=re.compile("^" + re.escape(model)))).first.click()
    page.wait_for_timeout(1200)
    page.get_by_role("button", name="Session options").click(); page.wait_for_timeout(1200)
    page.get_by_role("menuitem").filter(has_text="Reasoning effort").first.click(); page.wait_for_timeout(1000)
    page.locator("[role=menuitemradio], [role=option], [role=menuitem]").filter(has_text=re.compile("^" + effort)).first.click()
    page.wait_for_timeout(1000)
    page.keyboard.press("Escape"); page.wait_for_timeout(400)
    page.get_by_role("button", name="Session options").click(); page.wait_for_timeout(1200)
    eff = page.get_by_role("menuitem").filter(has_text="Reasoning effort").first.inner_text().split("\n")[1].strip()
    page.keyboard.press("Escape"); page.wait_for_timeout(600)
    got = page.get_by_role("button", name=re.compile(r"^Model: ")).last.get_attribute("aria-label")
    return got, eff


def busy(page):
    return page.get_by_role("button", name="Stop").count() > 0


def run_one(page, cond, nn, task, max_s=3600):
    stem = f"{nn:02d}_{task}"
    out = OUT / "runs" / cond / stem
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "cards.jsonl"
    def log(rec):
        with open(logf, "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")
    prompt = (PROMPTS / cond / f"{stem}.txt").read_text()
    page.goto("http://localhost:38765/"); page.wait_for_timeout(5000)
    page.get_by_role("button", name="New project").first.click(); page.wait_for_timeout(2500)
    dlg = page.locator("[role=dialog]").last
    dlg.get_by_placeholder("Project name").fill(f"{cond}_{stem}")
    dlg.get_by_role("button", name="Create").click()
    ed = page.get_by_role("textbox").first
    ed.wait_for(state="visible", timeout=180000)
    got, eff = set_model_effort(page)
    log({"event": "model_effort", "model": got, "effort": eff})
    if got != "Model: Sonnet 5" or not eff.startswith("Low"):
        raise RuntimeError(f"model/effort not pinned: {got} {eff}")
    ed = page.get_by_role("textbox").first
    ed.click(); ed.fill(prompt); page.wait_for_timeout(800)
    t0 = time.time()
    page.get_by_role("button", name="Send", exact=True).click()
    log({"event": "sent", "url": page.url})
    idle_since = None
    while time.time() - t0 < max_s:
        page.wait_for_timeout(10000)
        if handle_cards(page, log):
            idle_since = None
            continue
        if busy(page):
            idle_since = None
            continue
        idle_since = idle_since or time.time()
        if time.time() - idle_since < 45:
            continue
        break
    wall = time.time() - t0
    (out / "page.txt").write_text(page.locator("main").inner_text())
    page.screenshot(path=str(out / "page.png"), full_page=True)
    log({"event": "done", "wall_s": round(wall, 1), "url": page.url, "timed_out": wall >= max_s})
    return page.url, wall

