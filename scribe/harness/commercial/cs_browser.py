#!/usr/bin/env python3
"""Opens the Claude Science app in a persistent headless Chromium profile and signs in."""

import os, subprocess

DATA = os.environ["CS_DATA"]
PROFILE = os.environ["CS_PROFILE"]


def login_url() -> str:
    return subprocess.run(["claude-science", "url", "--data-dir", DATA],
                          capture_output=True, text=True, check=True).stdout.strip()


def open_app(p):
    ctx = p.chromium.launch_persistent_context(PROFILE, headless=True, viewport={"width": 1500, "height": 1000})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(login_url(), wait_until="domcontentloaded")
    page.wait_for_timeout(6000)
    return ctx, page


def login(p):
    ctx, page = open_app(p)
    b = page.get_by_role("button", name="Sign in")
    if b.count():
        b.first.click()
        page.wait_for_timeout(5000)
    return ctx, page
