"""Checks every cell of the four leaderboard tables in the paper's LaTeX source against the built
boards and exits non-zero on any mismatch. Usage: python check_paper_tables.py --tex <paper.tex>
--board-a <dir> --board-b <dir> --rows-a NAME=KEY,... --rows-b NAME=KEY,..."""

import argparse
import csv
import re
import sys

BASE_A = {'SurveyG': 'surveyg.ref', 'OpenAI tool loop': 'openai_luna_mcp.ref', 'SurveyGen-I': 'sgi.ref',
          'AutoSurvey': 'autosurvey.ref', 'LiRA': 'lira', 'LLM$\\times$MR': 'llmxmr.ref',
          'Claude Science': 'claude_science_mcp.ref', 'Claude Code': 'claude_sonnet_mcp.ref', 'Elicit SR$^{\\S}$': 'elicit_sr.ref'}
BASE_B = {'SurveyG': 'surveyg', 'OpenAI tool loop': 'openai_luna_mcp', 'SurveyGen-I': 'sgi', 'AutoSurvey': 'autosurvey',
          'SurveyForge': 'surveyforge', 'LLM$\\times$MR': 'llmxmr', 'Claude Science': 'claude_science_mcp',
          'Claude Code': 'claude_sonnet_mcp'}
OURS = ('SCRIBE', 'SCRIBE (untrained)', 'SCRIBE-Luna')
AXES_A = ['synthesis', 'reasoning', 'planning', 'writing', 'form']
AXES_B = ['retrieval', 'synthesis', 'reasoning', 'planning', 'writing', 'form']
UNREPORTED = 'gemini_web_dr.ref,gemini_web_dr'
TABLES = [('tab:leaderboard (a)', 'a', 'reference_fed', 'system_trunc', AXES_A),
          ('tab:leaderboard (b)', 'b', 'self_retrieving', 'system_trunc', AXES_B),
          ('tab:leaderboard-full (a)', 'a', 'reference_fed', 'system', AXES_A),
          ('tab:leaderboard-full (b)', 'b', 'self_retrieving', 'system', AXES_B)]
STRIP = re.compile(r'\\(?:rowcolor|cellcolor|arrayrulecolor)\{[^}]*\}')


def cell(raw):
    s = STRIP.sub('', raw).strip()
    m = re.fullmatch(r'\\(textbf|underline)\{(.*)\}', s)
    return (m.group(2).strip(), m.group(1)) if m else (s, None)


def rows(tex, label):
    m = re.search(r'% BEGIN ' + re.escape(label) + r'\n(.*?)% END ' + re.escape(label), tex, re.S)
    if m is None:
        raise SystemExit(f'{label}: no "% BEGIN/% END {label}" block in the tex')
    out, backbone = [], None
    for line in m.group(1).splitlines():
        if '&' not in line or 'Composite' in line or 'multicolumn' in line:
            continue
        raw = [c for c in re.sub(r'\\\\\s*$', '', line.strip()).split('&')]
        cells = [cell(c) for c in raw]
        backbone = cells[0][0] or backbone
        out.append({'backbone': backbone, 'name': cells[1][0], 'ours': '\\rowcolor{oursrow}' in line,
                    'values': [c[0] for c in cells[2:]], 'marks': [c[1] for c in cells[2:]]})
    return out


def rank_text(s):
    return s.replace('$', '').replace('--', '-')


def board(d, tier, obs):
    with open(f'{d}/summary.csv') as f:
        return {r['system']: r for r in csv.DictReader(f) if r['tier'] == tier and r['obs'] == obs}


def parse_rows(spec):
    return dict(x.split('=', 1) for x in (spec or '').split(',') if '=' in x)


def check_marks(label, trows, heads):
    bad = []
    for j, h in enumerate(heads):
        if h == 'rank':
            continue
        vals = sorted({float(r['values'][j]) for r in trows}, reverse=True)
        for r in trows:
            v = float(r['values'][j])
            want = 'textbf' if v == vals[0] else 'underline' if len(vals) > 1 and v == vals[1] else None
            if r['marks'][j] != want:
                bad.append(f'  MISMATCH {label} {r["name"]} {h}: marked {r["marks"][j]}, the column order gives {want}')
    return bad


def check_table(label, trows, B, names, axes, unreported):
    bad = []
    heads = ['composite', 'rank'] + axes
    seen = set()
    for r in trows:
        key = names.get(r['name'])
        if key is None:
            bad.append(f'  UNMAPPED {label} {r["name"]}: give its board key with --rows-a/--rows-b')
            continue
        if key not in B:
            bad.append(f'  MISSING {label} {r["name"]}: {key} is not on the board')
            continue
        seen.add(key)
        b = B[key]
        want = [f"{float(b['composite']):.3f}", b['cert_interval']] + [f"{float(b['z_' + x]):.3f}" for x in axes]
        got = [r['values'][0], rank_text(r['values'][1])] + r['values'][2:]
        if len(got) != len(want):
            bad.append(f'  MISMATCH {label} {r["name"]}: {len(got)} cells, the board has {len(want)}')
            continue
        for g, w, h in zip(got, want, heads):
            if g != w:
                bad.append(f'  MISMATCH {label} {r["name"]} {h}: tex={g} board={w}')
        bb = (b.get('backbone') or '').split(' ')[0]
        if r['backbone'] != bb:
            bad.append(f'  MISMATCH {label} {r["name"]} backbone: tex={r["backbone"]} board={b.get("backbone")!r}')
        if r['ours'] != (b.get('role') == 'ours'):
            bad.append(f'  MISMATCH {label} {r["name"]} shading: tex ours={r["ours"]} board role={b.get("role")!r}')
    extra = sorted(set(B) - seen - set(unreported))
    if extra:
        bad.append(f'  MISMATCH {label}: board systems {extra} are neither in the table nor declared unreported')
    return bad + check_marks(label, trows, heads), len(trows) * (len(heads) + 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', required=True)
    ap.add_argument('--board-a', required=True, help='board directory of the fixed-input tables')
    ap.add_argument('--board-b', required=True, help='board directory of the same-pool tables')
    ap.add_argument('--rows-a', required=True, help='NAME=KEY,... board keys of SCRIBE, SCRIBE (untrained) and SCRIBE-Luna under fixed input')
    ap.add_argument('--rows-b', required=True, help='NAME=KEY,... board keys of SCRIBE, SCRIBE (untrained) and SCRIBE-Luna under same pool')
    a = ap.parse_args(argv)
    with open(a.tex) as f:
        tex = f.read()
    maps = {'a': {**BASE_A, **parse_rows(a.rows_a)}, 'b': {**BASE_B, **parse_rows(a.rows_b)}}
    missing = [f'--rows-{p} {n}' for p in 'ab' for n in OURS if n not in maps[p]]
    if missing:
        raise SystemExit(f'no board key for {missing}')
    dirs = {'a': a.board_a, 'b': a.board_b}
    unreported = [s for s in UNREPORTED.split(',') if s]
    nbad = 0
    for label, p, tier, obs, axes in TABLES:
        bad, ncell = check_table(label, rows(tex, label), board(dirs[p], tier, obs), maps[p], axes, unreported)
        for line in bad:
            print(line)
        nbad += len(bad)
        print(f'{label} vs {dirs[p]} {tier}/{obs}: {ncell} cells, {len(bad)} mismatches')
    print(f'all tables: {nbad} mismatches')
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main())
