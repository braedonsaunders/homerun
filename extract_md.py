"""Extract the Homerun tutorial content from homerun_body.html into a clean
Markdown article. Stdlib only (no bs4)."""
import re
import html as _html
from html.parser import HTMLParser

SRC = open(r'C:\Users\bsaun\Downloads\homerun-tutorial.html', encoding='utf-8').read()
OUT = r'C:\Users\bsaun\Downloads\homerun-tutorial.md'

# Strip non-content: inlined CSS, SVG diagrams, scripts.
clean = re.sub(r'<style>.*?</style>', '', SRC, flags=re.S)
clean = re.sub(r'<svg.*?</svg>', '', clean, flags=re.S)
clean = re.sub(r'<script.*?</script>', '', clean, flags=re.S)

VOID = {'img', 'br', 'hr', 'meta', 'link', 'input', 'source'}


class Node:
    __slots__ = ('tag', 'attrs', 'children', 'parent')

    def __init__(self, tag, attrs):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []
        self.parent = None

    @property
    def toks(self):
        return set(self.attrs.get('class', '').split())

    def descendants(self):
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.descendants()

    def first(self, token):
        for d in self.descendants():
            if token in d.toks:
                return d
        return None

    def first_tag(self, tag):
        for d in self.descendants():
            if d.tag == tag:
                return d
        return None


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node('root', {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        n.parent = self.cur
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        n = Node(tag, attrs)
        n.parent = self.cur
        self.cur.children.append(n)

    def handle_endtag(self, tag):
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root and node.parent is not None:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def inline(node):
    """Render inline children to Markdown."""
    parts = []
    for c in node.children:
        if isinstance(c, str):
            parts.append(c)
            continue
        t = c.tag
        if t in ('strong', 'b'):
            parts.append('**' + inline(c).strip() + '**')
        elif t in ('em', 'i'):
            parts.append('*' + inline(c).strip() + '*')
        elif t == 'code':
            parts.append('`' + inline(c).strip() + '`')
        elif t == 'a':
            parts.append('[' + inline(c).strip() + '](' + c.attrs.get('href', '') + ')')
        elif t == 'br':
            parts.append(' ')
        else:
            parts.append(inline(c))
    return ''.join(parts)


def para(node):
    return re.sub(r'\s+', ' ', inline(node)).strip()


def raw(node):
    """Concatenate raw text (for <pre>), preserving newlines."""
    parts = []
    for c in node.children:
        if isinstance(c, str):
            parts.append(c)
        else:
            parts.append(raw(c))
    return ''.join(parts).strip('\n')


tree = Tree()
tree.feed(clean)

md = []

# ── Title + meta ──
h1 = re.search(r'<h1>(.*?)</h1>', SRC, re.S)
title = re.sub(r'<[^>]+>', '', _html.unescape(h1.group(1))).strip() if h1 else 'Homerun Tutorial'
md.append('# ' + title)
md.append('*A 7-step guide to going from idea to live bot on Homerun · ~35 minutes*')
md.append('')
md.append('> 📸 **Screenshot idea:** hero / dashboard overview shot to open the article.')

SHOT_IDEAS = {
    2: "Accounts → Sandbox Desk creating a sandbox account, plus the global account selector in the top control bar.",
    4: "The Opportunities scanner in Cards view — a single opportunity card showing cost, expected payout, confidence, and the guaranteed-arbitrage badge.",
    5: "The Datasets workbench browsing microstructure snapshots / recording sessions, and the Sources view with the Source Kind picker + “Generate with AI”.",
    6: "A Research backtest result — the equity curve, the pessimistic/realistic/optimistic fill bands, and the backtest-vs-shadow-vs-live triangulation chart.",
    7: "The Bots roster with the + (New bot) button, and the bot configuration flyout (strategy picker, risk limits, schedule).",
}


def emit_prose(node):
    for c in node.children:
        if not isinstance(c, Node):
            continue
        if c.tag == 'h3':
            md.append('### ' + para(c))
            md.append('')
        elif c.tag == 'p':
            md.append(para(c))
            md.append('')
        elif c.tag == 'ul':
            for li in c.children:
                if isinstance(li, Node) and li.tag == 'li':
                    md.append('- ' + para(li))
            md.append('')


def emit_ai(node):
    title_el = node.first('block-ai-title')
    ctx = node.first('block-ai-context')
    prompt = node.first('block-ai-prompt')
    t = para(title_el) if title_el else 'Follow along with AI'
    md.append('> **🤖 Follow along with AI — ' + t + '**')
    if ctx:
        md.append('>')
        md.append('> ' + para(ctx))
    md.append('')
    if prompt:
        md.append('```text')
        md.append(raw(prompt))
        md.append('```')
        md.append('')


def emit_code(node):
    lang_el = node.first('lang-chip')
    fn_el = node.first('filename')
    cap = node.first('block-code-caption')
    pre_el = node.first_tag('pre')  # the real code lives in <pre><code>, not the caption's inline <code>
    code_el = pre_el.first_tag('code') if pre_el else None
    lang = para(lang_el) if lang_el else ''
    if cap:
        c = para(cap)
        if fn_el:
            c += '  (`' + para(fn_el) + '`)'
        md.append('*' + c + '*')
        md.append('')
    md.append('```' + lang)
    md.append(raw(code_el) if code_el else '')
    md.append('```')
    md.append('')


def emit_screenshot(node):
    img = node.first_tag('img')
    cap = node.first('block-screenshot-caption')
    if img:
        md.append('![' + img.attrs.get('alt', '') + '](' + img.attrs.get('src', '') + ')')
        md.append('')
    if cap:
        md.append('*' + para(cap) + '*')
        md.append('')


def emit_embed(node):
    cap = node.first('block-embed-caption')
    captext = para(cap) if cap else ''
    low = captext.lower()
    if 'six-stage pipeline' in low or 'pipeline' in low:
        md.append('```')
        md.append('Detect  →  Evaluate  →  Preflight  →  Arm  →  Execute  →  Monitor')
        md.append('```')
    elif 'headline metrics' in low or 'illustrative headline' in low:
        md.append('| ROI (net of fees) | Sharpe | Trades | Max drawdown |')
        md.append('|---|---|---|---|')
        md.append('| +12.7% | 1.18 | 41 | -7.9% |')
    elif 'global switch' in low or 'sandbox account is shadow' in low:
        md.append('```')
        md.append('Sandbox account (shadow, $0 risk)')
        md.append('        │')
        md.append('   global account switch  ──  live = live preflight + arm gate')
        md.append('        ▼')
        md.append('Live account (real CLOB orders)')
        md.append('```')
    if captext:
        md.append('')
        md.append('*' + captext + '*')
    md.append('')


def emit_callout(node):
    lbl = node.first('lbl')
    h4 = node.first_tag('h4')
    head = ''
    if lbl:
        head += para(lbl)
    if h4:
        head += (' — ' if head else '') + para(h4)
    md.append('> **' + head + '**')
    for p in node.children:
        if isinstance(p, Node) and p.tag == 'p':
            md.append('>')
            md.append('> ' + para(p))
    md.append('')


def emit_intro_card(node):
    head = node.first('intro-card-head')
    if head:
        md.append('**' + para(head) + '**')
        md.append('')
    ul = node.first_tag('ul')
    if ul:
        for li in ul.children:
            if isinstance(li, Node) and li.tag == 'li':
                md.append('- ' + para(li))
        md.append('')


def walk(node, page_index):
    for c in node.children:
        if not isinstance(c, Node):
            continue
        toks = c.toks
        if 'step-heading' in toks:
            md.append('## Step %d: %s' % (page_index, para(c)))
            md.append('')
        elif 'step-intro' in toks:
            md.append('_' + ' '.join(para(p) for p in c.children if isinstance(p, Node) and p.tag == 'p') + '_')
            md.append('')
        elif 'intro-lede' in toks:
            md.append(para(c))
            md.append('')
        elif 'intro-card' in toks:
            emit_intro_card(c)
        elif 'block-prose' in toks:
            emit_prose(c)
        elif 'block-ai' in toks:
            emit_ai(c)
        elif 'block-code' in toks:
            emit_code(c)
        elif 'block-screenshot' in toks:
            emit_screenshot(c)
        elif 'block-embed' in toks:
            emit_embed(c)
        elif 'block-callout' in toks:
            emit_callout(c)
        else:
            walk(c, page_index)


sections = [d for d in tree.root.descendants()
            if d.tag == 'section' and 'tut-page' in d.toks]
for sec in sections:
    idx = int(sec.attrs.get('data-page-index', '0'))
    if idx == 0:
        md.append('')
        md.append('---')
        md.append('')
    else:
        md.append('')
        md.append('---')
        md.append('')
    walk(sec, idx)
    if idx in SHOT_IDEAS:
        md.append('> 📸 **Screenshot idea:** ' + SHOT_IDEAS[idx])
        md.append('')

text = '\n'.join(md)
# tidy: collapse 3+ blank lines
text = re.sub(r'\n{3,}', '\n\n', text)
open(OUT, 'w', encoding='utf-8').write(text)
print('wrote', OUT)
print('lines:', text.count('\n') + 1, 'chars:', len(text))
