"""Build verifier_model_case.docx from the markdown with a layout that holds in Word, Pages and Google Docs."""
import re, sys, zipfile, shutil, math
import pypandoc, docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC, RAW, OUT = sys.argv[1], "/root/vbench/vmc_raw.docx", sys.argv[2]

# --- markdown: drop the H1 (it becomes the title), turn the Contents heading into a plain label
md = open(SRC).read().split("\n")
assert md[0].startswith("# ")
md = "\n".join(md[1:])
md = md.replace("\n## Contents\n", "\n**Contents**\n", 1)
md = re.sub(r"(\*\*Contents\*\*\n.*?)\n---\n", r"\1\n", md, count=1, flags=re.S)
pass
assert "\n---\n" not in md
open("/root/vbench/vmc_docx.md", "w").write(md)

pypandoc.convert_file("/root/vbench/vmc_docx.md", "docx", outputfile=RAW,
    extra_args=["--from=markdown+tex_math_dollars+pipe_tables",
                "--metadata=title:The case for a verifier model",
                "--metadata=subtitle:Why agents write bad graders, what fixes it, and why HUD should build a verifier model"])

d = docx.Document(RAW)
body = d.element.body

# --- page: US Letter, 1 inch sides, 0.9 inch top and bottom
PAGE_W, PAGE_H, SIDE, TB = 12240, 15840, 1440, 1150
TEXT_W = PAGE_W - 2 * SIDE
sect = body.find(qn("w:sectPr"))
for tag in ("w:pgSz", "w:pgMar"):
    for old in sect.findall(qn(tag)):
        sect.remove(old)
pgSz = OxmlElement("w:pgSz"); pgSz.set(qn("w:w"), str(PAGE_W)); pgSz.set(qn("w:h"), str(PAGE_H))
pgMar = OxmlElement("w:pgMar")
for k, v in (("top", TB), ("right", SIDE), ("bottom", TB), ("left", SIDE), ("header", 700), ("footer", 700), ("gutter", 0)):
    pgMar.set(qn(f"w:{k}"), str(v))
sect.insert(0, pgMar); sect.insert(0, pgSz)

# --- page numbers in the footer
from docx.enum.section import WD_SECTION
sec = d.sections[0]
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
def field(par, instr):
    r = par.add_run()
    for kind, text in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        if kind:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._r.append(fc)
        elif text == instr:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "; r._r.append(it)
        else:
            t = OxmlElement("w:t"); t.text = text; r._r.append(t)
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
field(fp, "PAGE")

# --- styles
st = {s.name: s for s in d.styles}
BLUE = RGBColor(0x1F, 0x4E, 0x79)
for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
    if name in st:
        st[name].font.name = "Calibri"; st[name].font.size = Pt(10.5)
if "Compact" in st:
    st["Compact"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
for name in ("Body Text", "First Paragraph"):
    if name in st:
        pf = st[name].paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_before = Pt(0); pf.space_after = Pt(6); pf.line_spacing = 1.08
for name, size, before, after in (("Title", 24, 0, 4), ("Subtitle", 13, 0, 14), ("Heading 1", 16, 18, 6),
                                  ("Heading 2", 14.5, 18, 6), ("Heading 3", 12, 12, 4)):
    if name in st:
        s = st[name]; s.font.name = "Calibri"; s.font.size = Pt(size); s.font.color.rgb = BLUE; s.font.bold = True
        s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
        if name in ("Title", "Subtitle"):
            s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
if "Subtitle" in st:
    st["Subtitle"].font.bold = False; st["Subtitle"].font.color.rgb = RGBColor(0x40, 0x40, 0x40)

# --- tables
def text_len(tc):
    return len("".join(t.text or "" for t in tc.iter(qn("w:t"))))

def set_tbl_props(tbl, widths):
    tblPr = tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout", "w:tblBorders", "w:tblCellMar", "w:jc", "w:tblInd", "w:tblLook"):
        for old in tblPr.findall(qn(tag)):
            tblPr.remove(old)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), str(TEXT_W)); w.set(qn("w:type"), "dxa"); tblPr.append(w)
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblPr.append(jc)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "A6A6A6"); b.append(e)
    tblPr.append(b)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", 50), ("left", 100), ("bottom", 50), ("right", 100)):
        m = OxmlElement(f"w:{side}"); m.set(qn("w:w"), str(v)); m.set(qn("w:type"), "dxa"); mar.append(m)
    tblPr.append(mar)
    grid = tbl.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for wd in widths:
        gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(wd)); grid.append(gc)

def shade(tc, fill):
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill)
    tcPr.append(s)

def set_tcw(tc, wd):
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcW")):
        tcPr.remove(old)
    w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(wd)); w.set(qn("w:type"), "dxa"); tcPr.insert(0, w)

def keep_next(p, on=True):
    p.paragraph_format.keep_with_next = True if on else None

CH = 105   # dxa per character of 9 pt Calibri, generous
PAD = 230  # cell margins plus slack

def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(qn("w:t")))

def widths_for(rows):
    ncol = max(len(r) for r in rows)
    mins, wants = [], []
    for c in range(ncol):
        texts = [cell_text(r[c]) for r in rows if c < len(r)]
        longest_word = max((len(w) for t in texts for w in re.split(r"[\s-]+", t) if w), default=4)
        longest = max((len(t) for t in texts), default=4)
        mins.append(min(longest_word, 22) * CH + PAD)
        wants.append(min(longest, 55) * CH + PAD)
    mins = [max(m, 700) for m in mins]
    wants = [max(w, m) for w, m in zip(wants, mins)]
    if sum(wants) <= TEXT_W:
        extra = TEXT_W - sum(wants)
        w = [x + extra * x / sum(wants) for x in wants]
    elif sum(mins) >= TEXT_W:
        w = [x * TEXT_W / sum(mins) for x in mins]
    else:
        room = TEXT_W - sum(mins)
        slack = [x - m for x, m in zip(wants, mins)]
        w = [m + room * sl / sum(slack) for m, sl in zip(mins, slack)]
    w = [int(x) for x in w]
    w[0] += TEXT_W - sum(w)
    return w

ntables = 0
for tbl in body.iter(qn("w:tbl")):
    ntables += 1
    trs = tbl.findall(qn("w:tr"))
    rows = [tr.findall(qn("w:tc")) for tr in trs]
    widths = widths_for(rows)
    set_tbl_props(tbl, widths)
    for i, (tr, cells) in enumerate(zip(trs, rows)):
        trPr = tr.find(qn("w:trPr"))
        if trPr is None:
            trPr = OxmlElement("w:trPr"); tr.insert(0, trPr)
        for tag in ("w:cantSplit", "w:tblHeader"):
            for old in trPr.findall(qn(tag)):
                trPr.remove(old)
        trPr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            h = OxmlElement("w:tblHeader"); trPr.append(h)
        for c, tc in enumerate(cells):
            set_tcw(tc, widths[c] if c < len(widths) else widths[-1])
            if i == 0:
                shade(tc, "1F4E79")
            elif i % 2 == 0:
                shade(tc, "EEF3F8")
            for pe in tc.iter(qn("w:p")):
                p = docx.text.paragraph.Paragraph(pe, None)
                p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                if pe.pPr is None or pe.pPr.find(qn("w:jc")) is None:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                # keep the whole table on one page: every row but the last holds on to the next
                keep_next(p, i < len(trs) - 1)
                for r in p.runs:
                    r.font.size = Pt(9)
                    if i == 0:
                        r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                if i == 0:
                    for mr in pe.iter(qn("m:r")):
                        rpr = mr.find(qn("w:rPr"))
                        if rpr is None:
                            rpr = OxmlElement("w:rPr"); mr.insert(0, rpr)
                        col = OxmlElement("w:color"); col.set(qn("w:val"), "FFFFFF"); rpr.append(col)
    # a short lead-in line ending in ':' stays with its table; so does a heading straight above it
    prev = tbl.getprevious()
    if prev is not None and prev.tag == qn("w:p"):
        txt = "".join(t.text or "" for t in prev.iter(qn("w:t"))).strip()
        if txt.endswith(":") or len(txt) < 140:
            keep_next(docx.text.paragraph.Paragraph(prev, None))
    # space after the table
    nxt = tbl.getnext()
    if nxt is not None and nxt.tag == qn("w:p"):
        docx.text.paragraph.Paragraph(nxt, None).paragraph_format.space_before = Pt(6)

# display equations: centred, kept with the table or text they introduce when short
for pe in body.iter(qn("w:p")):
    if pe.find(qn("m:oMathPara")) is not None:
        p = docx.text.paragraph.Paragraph(pe, None)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)

# paragraphs carrying a long link read badly justified; the closing sources note is small print
for pe in body.iter(qn("w:p")):
    txt = "".join(t.text or "" for t in pe.iter(qn("w:t")))
    p = docx.text.paragraph.Paragraph(pe, None)
    if "http" in txt and pe.getparent().tag == qn("w:body"):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if txt.startswith("Sources:"):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(8)
        for r in p.runs:
            r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

d.save(OUT)

# --- leave Word's compatibility mode: declare the current layout engine
tmp = OUT + ".tmp"
zin = zipfile.ZipFile(OUT); zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/settings.xml":
        x = data.decode()
        x = re.sub(r"<w:compat>.*?</w:compat>|<w:compat/>", "", x, flags=re.S)
        compat = ('<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/>'
                  '<w:compatSetting w:name="overrideTableStyleFontSizeAndJustification" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/>'
                  '<w:compatSetting w:name="enableOpenTypeFeatures" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/>'
                  '<w:compatSetting w:name="doNotFlipMirrorIndents" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/></w:compat>')
        x = x.replace("</w:settings>", compat + "</w:settings>")
        x = re.sub(r"<w:updateFields[^>]*/>", "", x)
        data = x.encode()
    zout.writestr(item, data)
zin.close(); zout.close(); shutil.move(tmp, OUT)

x = zipfile.ZipFile(OUT).read("word/document.xml").decode()
print("tables", ntables, "display eqs", x.count("<m:oMathPara>"), "fields", x.count("fldChar"),
      "dashes", len(re.findall("[–—]", x)), "toc", x.count("TOC \\"))
