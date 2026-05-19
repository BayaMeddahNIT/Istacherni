#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_arch_pdf.py
Converts the Istacherni Architecture Roadmap markdown to a styled PDF.
Uses fpdf2 with Windows Arial TTF for full Unicode support.
"""

import re
import sys
from pathlib import Path
from fpdf import FPDF

# ── Paths ──────────────────────────────────────────────────────────────────────
MD_FILE  = Path(r"C:\Users\pc cam\.gemini\antigravity\brain\64baa725-f3a1-4fd7-9ed7-073d2997b5d3\Istacherni_Full_Architecture_Roadmap.md")
OUT_FILE = Path(r"C:\Users\pc cam\Desktop\L3\pfe baya\Istacherni\Istacherni_Architecture_Roadmap.pdf")

ARIAL_R  = r"C:\Windows\Fonts\arial.ttf"
ARIAL_B  = r"C:\Windows\Fonts\arialbd.ttf"
ARIAL_I  = r"C:\Windows\Fonts\ariali.ttf"
ARIAL_BI = r"C:\Windows\Fonts\arialbi.ttf"
COUR_R   = r"C:\Windows\Fonts\cour.ttf"   # Courier New Regular

# ── Colours (R, G, B) ─────────────────────────────────────────────────────────
C_BG        = (15,  17,  35)
C_CARD      = (26,  30,  55)
C_H1        = (99, 179, 237)
C_H2        = (129, 230, 217)
C_H3        = (246, 173,  85)
C_BODY      = (226, 232, 240)
C_CODE_BG   = (20,  24,  45)
C_CODE_FG   = (190, 242, 100)
C_TABLE_H   = (44,  82, 130)
C_TABLE_ALT = (22,  27,  50)
C_RULE      = (74,  85, 104)
C_ACCENT    = (237, 100, 166)

MARGIN   = 15
PAGE_W   = 210
USABLE_W = PAGE_W - 2 * MARGIN


def safe(text: str) -> str:
    """Replace characters that cause encoding issues with safe equivalents."""
    MAP = {
        '\u2192': '->',   # →
        '\u2190': '<-',   # ←
        '\u2193': 'v',    # ↓
        '\u2191': '^',    # ↑
        '\u2022': '*',    # •
        '\u2013': '-',    # –
        '\u2014': '--',   # —
        '\u201c': '"',    # "
        '\u201d': '"',    # "
        '\u2018': "'",    # '
        '\u2019': "'",    # '
        '\u2026': '...',  # …
        '\u2264': '<=',   # ≤
        '\u2265': '>=',   # ≥
        '\u2260': '!=',   # ≠
        '\u00d7': 'x',    # ×
        '\u2211': 'sum',  # ∑
        '\u2208': 'in',   # ∈
        '\u2229': 'AND',  # ∩
        '\u2200': 'A',    # ∀
        '\u03b1': 'alpha',# α
        '\u03b2': 'beta', # β
        '\u03c3': 'sigma',# σ
        '\u03b8': 'theta',# θ
        '\u22c5': '.',    # ⋅
        '\u221e': 'inf',  # ∞
        '\u2248': '~=',   # ≈
        '\u0307': '',     # combining dot (remove)
        '\u00b7': '.',    # ·
        '\u2016': '||',   # ‖
    }
    for orig, repl in MAP.items():
        text = text.replace(orig, repl)
    # Strip Arabic/RTL characters (U+0600–U+06FF) and other non-latin-1
    # Replace Arabic runs with [Arabic text]
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', '[ar]', text)
    # Remove any remaining non-latin-1 chars
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


class ArchPDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        # Register Unicode-capable fonts
        self.add_font('Arial',   '',  ARIAL_R)
        self.add_font('Arial',   'B', ARIAL_B)
        self.add_font('Arial',   'I', ARIAL_I)
        self.add_font('Arial',   'BI', ARIAL_BI)
        if Path(COUR_R).exists():
            self.add_font('Courier', '', COUR_R)
            self._has_courier = True
        else:
            self._has_courier = False

    # ── FPDF callbacks ──────────────────────────────────────────────────────────
    def header(self):
        self.set_fill_color(*C_BG)
        self.rect(0, 0, 210, 297, 'F')
        self.set_fill_color(*C_H1)
        self.rect(0, 0, 210, 1.5, 'F')

    def footer(self):
        self.set_y(-12)
        self.set_font('Arial', 'I', 7)
        self.set_text_color(*C_RULE)
        self.cell(0, 5, f'Istacherni - Hybrid RAG Architecture Roadmap    |    Page {self.page_no()}', align='C')

    # ── Emit helpers ────────────────────────────────────────────────────────────
    def rule(self):
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.3)
        y = self.get_y() + 1
        self.line(MARGIN, y, PAGE_W - MARGIN, y)
        self.ln(3)

    def emit_h1(self, text):
        self.ln(4)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*C_H1)
        self.multi_cell(USABLE_W, 8, safe(text), align='L')
        self.rule()
        self.ln(1)

    def emit_h2(self, text):
        self.ln(3)
        self.set_fill_color(*C_H2)
        self.rect(MARGIN, self.get_y(), 2, 6, 'F')
        self.set_x(MARGIN + 4)
        self.set_font('Arial', 'B', 13)
        self.set_text_color(*C_H2)
        self.multi_cell(USABLE_W - 4, 6, safe(text), align='L')
        self.ln(1)

    def emit_h3(self, text):
        self.ln(2)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(*C_H3)
        self.multi_cell(USABLE_W, 5.5, safe(text), align='L')
        self.ln(0.5)

    def emit_body(self, text):
        self.set_font('Arial', '', 9)
        self.set_text_color(*C_BODY)
        self.multi_cell(USABLE_W, 5, safe(text), align='L')

    def emit_italic(self, text):
        self.set_font('Arial', 'I', 8.5)
        self.set_text_color(*C_H3)
        self.multi_cell(USABLE_W, 4.8, safe(text), align='C')
        self.ln(1)
        self.set_text_color(*C_BODY)

    def emit_code_block(self, lines):
        pad = 3
        line_h = 4.2
        block_h = len(lines) * line_h + pad * 2
        if self.get_y() + block_h > 270:
            self.add_page()
        self.set_fill_color(*C_CODE_BG)
        self.set_draw_color(*C_H2)
        self.set_line_width(0.2)
        x, y = MARGIN, self.get_y()
        self.rect(x, y, USABLE_W, block_h, 'DF')
        self.set_xy(x + pad + 1, y + pad)
        font = 'Courier' if self._has_courier else 'Arial'
        self.set_font(font, '', 7.2)
        self.set_text_color(*C_CODE_FG)
        for line in lines:
            self.set_x(x + pad + 1)
            self.cell(USABLE_W - pad * 2, line_h, safe(line[:140]), ln=1)
        self.ln(2)
        self.set_text_color(*C_BODY)

    def emit_table(self, rows):
        if not rows:
            return
        n_cols = max(len(r) for r in rows)
        col_w  = USABLE_W / n_cols
        for r_idx, row in enumerate(rows):
            if self.get_y() + 6 > 270:
                self.add_page()
            if r_idx == 0:
                self.set_fill_color(*C_TABLE_H)
                self.set_font('Arial', 'B', 8)
                self.set_text_color(255, 255, 255)
            elif r_idx % 2 == 0:
                self.set_fill_color(*C_TABLE_ALT)
                self.set_font('Arial', '', 7.5)
                self.set_text_color(*C_BODY)
            else:
                self.set_fill_color(*C_CARD)
                self.set_font('Arial', '', 7.5)
                self.set_text_color(*C_BODY)
            # pad row to n_cols
            row_padded = row + [''] * (n_cols - len(row))
            for cell in row_padded:
                self.cell(col_w, 5.5, safe(str(cell))[:60], border=0, fill=True)
            self.ln(5.5)
        self.ln(2)

    def emit_bullet(self, text, level=0):
        indent = MARGIN + level * 4
        bullet  = '*' if level == 0 else '-'
        self.set_font('Arial', '', 9)
        self.set_text_color(*C_BODY)
        self.set_x(indent + 3)
        self.cell(4, 5, bullet)
        self.set_x(indent + 7)
        self.multi_cell(USABLE_W - indent - 7 + MARGIN, 5, safe(text.strip()), align='L')

    def emit_blockquote(self, text):
        self.set_fill_color(*C_CARD)
        self.rect(MARGIN, self.get_y(), USABLE_W, 6, 'F')
        self.set_fill_color(*C_H2)
        self.rect(MARGIN, self.get_y(), 2, 6, 'F')
        self.set_x(MARGIN + 4)
        self.set_font('Arial', 'I', 8.5)
        self.set_text_color(*C_H2)
        self.multi_cell(USABLE_W - 4, 6, safe(text), align='L')
        self.ln(1)
        self.set_text_color(*C_BODY)

    def cover_page(self):
        self.add_page()
        # Banner card
        self.set_fill_color(*C_CARD)
        self.rect(MARGIN, 40, USABLE_W, 90, 'F')
        self.set_fill_color(*C_H1)
        self.rect(MARGIN, 40, 4, 90, 'F')

        self.set_xy(MARGIN + 10, 50)
        self.set_font('Arial', 'B', 28)
        self.set_text_color(*C_H1)
        self.multi_cell(USABLE_W - 12, 13, 'Istacherni', align='L')

        self.set_x(MARGIN + 10)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*C_H2)
        self.multi_cell(USABLE_W - 12, 8, 'Hybrid RAG Architecture Roadmap', align='L')

        self.ln(3)
        self.set_x(MARGIN + 10)
        self.set_font('Arial', '', 10)
        self.set_text_color(*C_BODY)
        self.multi_cell(USABLE_W - 12, 6,
            'Complete Pipeline:\n'
            'Data Import -> BM25 -> BGE-M3 -> RRF Fusion\n'
            '-> Cross-Encoder Reranking -> Gemma2:9b -> Dual Evaluation',
            align='L')

        self.set_xy(MARGIN + 10, 148)
        self.set_font('Arial', '', 9)
        self.set_text_color(*C_H3)
        self.multi_cell(USABLE_W - 12, 6,
            'Algerian Legal QA System\n'
            'Fine-tuned BGE-M3 + BAAI/bge-reranker-v2-m3 + Gemma2:9b via Ollama\n'
            'P@1 = 82.68%  |  MRR = 0.8731  |  Hit@5 = 94.49%  |  NDCG@5 = 0.8910',
            align='L')

        # Stats bar
        self.set_fill_color(*C_TABLE_H)
        self.rect(MARGIN, 195, USABLE_W, 18, 'F')
        self.set_xy(MARGIN + 4, 199)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.cell(USABLE_W / 4, 6, 'Stages: 9',     align='C')
        self.cell(USABLE_W / 4, 6, 'P@1: 82.68%',   align='C')
        self.cell(USABLE_W / 4, 6, 'MRR: 0.8731',   align='C')
        self.cell(USABLE_W / 4, 6, 'alpha*: 0.70',  align='C')

        self.set_xy(MARGIN, 260)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*C_RULE)
        self.cell(USABLE_W, 5, 'Generated from source code analysis - May 2026', align='R')


# ── Markdown parser ────────────────────────────────────────────────────────────

def parse_table(lines):
    rows = []
    for line in lines:
        if re.match(r'\s*\|[-| :]+\|\s*$', line):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells:
            rows.append(cells)
    return rows


def strip_md(text: str) -> str:
    """Strip common markdown markup for plain rendering."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'`(.*?)`',       r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text


def render(pdf: ArchPDF, md_text: str):
    lines   = md_text.splitlines()
    i       = 0
    in_code = False
    code_buf= []
    in_table= False
    table_buf=[]

    while i < len(lines):
        raw      = lines[i]
        stripped = raw.strip()

        # ── code block ────────────────────────────────────────────────────────
        if stripped.startswith('```'):
            if not in_code:
                in_code  = True
                code_buf = []
            else:
                pdf.emit_code_block(code_buf)
                in_code  = False
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(raw.expandtabs(4))
            i += 1
            continue

        # ── table ─────────────────────────────────────────────────────────────
        if stripped.startswith('|'):
            if not in_table:
                in_table  = True
                table_buf = []
            table_buf.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                pdf.emit_table(parse_table(table_buf))
                in_table  = False
                table_buf = []

        # ── horizontal rule ───────────────────────────────────────────────────
        if re.match(r'^---+$', stripped):
            pdf.rule()
            i += 1
            continue

        # ── headings ─────────────────────────────────────────────────────────
        if stripped.startswith('# ') and not stripped.startswith('##'):
            pdf.emit_h1(stripped[2:])
            i += 1
            continue
        if stripped.startswith('## '):
            pdf.emit_h2(stripped[3:])
            i += 1
            continue
        if stripped.startswith('### '):
            pdf.emit_h3(stripped[4:])
            i += 1
            continue
        if stripped.startswith('#### '):
            pdf.emit_h3(stripped[5:])
            i += 1
            continue

        # ── math blocks ───────────────────────────────────────────────────────
        if stripped.startswith('$$'):
            math_lines = [stripped]
            if stripped.count('$$') < 2:
                i += 1
                while i < len(lines):
                    math_lines.append(lines[i].strip())
                    if '$$' in lines[i]:
                        break
                    i += 1
            pdf.emit_italic(' '.join(math_lines))
            i += 1
            continue

        # ── blockquote ────────────────────────────────────────────────────────
        if stripped.startswith('>'):
            text = stripped.lstrip('> ').strip()
            pdf.emit_blockquote(strip_md(text))
            i += 1
            continue

        # ── bullets ───────────────────────────────────────────────────────────
        m = re.match(r'^(\s*)[-*]\s+(.*)', raw)
        if m:
            level = len(m.group(1)) // 2
            text  = strip_md(m.group(2))
            pdf.emit_bullet(text, level)
            i += 1
            continue

        # ── numbered list ─────────────────────────────────────────────────────
        m2 = re.match(r'^(\s*)\d+\.\s+(.*)', raw)
        if m2:
            level = len(m2.group(1)) // 2
            text  = strip_md(m2.group(2))
            pdf.emit_bullet(text, level)
            i += 1
            continue

        # ── blank line ────────────────────────────────────────────────────────
        if stripped == '':
            pdf.ln(2)
            i += 1
            continue

        # ── normal body ───────────────────────────────────────────────────────
        pdf.emit_body(strip_md(stripped))
        i += 1

    # flush remaining table
    if in_table and table_buf:
        pdf.emit_table(parse_table(table_buf))


def main():
    if not MD_FILE.exists():
        print(f"ERROR: Markdown file not found:\n  {MD_FILE}")
        sys.exit(1)

    md_text = MD_FILE.read_text(encoding='utf-8')

    pdf = ArchPDF()
    pdf.cover_page()
    pdf.add_page()
    render(pdf, md_text)
    pdf.output(str(OUT_FILE))
    print(f"[OK] PDF saved to:\n  {OUT_FILE}")


if __name__ == '__main__':
    main()
