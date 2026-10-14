#!/usr/bin/env python3
"""Checks for the offline website mirror task.

Expected content set = what `wget -r -l 5 -k -X /admin -R zip -nH` produces from
http://localhost:8080/ serving /var/www/html (recomputed on the task image).
"""
import os, re, sys, posixpath

MIRROR = "/home/user/website_mirror"
SRC = "/var/www/html"

EXPECTED = """api-reference/authentication.html api-reference/endpoints.html api-reference/errors.html
api-reference/index.html blog/index.html blog/post1.html blog/post2.html css/main.css
docs/configuration/advanced.html docs/configuration/basic.html docs/configuration/index.html
docs/getting-started.html docs/index.html docs/installation/advanced/clustering.html
docs/installation/advanced/index.html docs/installation/advanced/security/deep/index.html
docs/installation/advanced/security/index.html docs/installation/advanced/security/ssl.html
docs/installation/index.html docs/installation/linux.html docs/installation/windows.html
images/logo.png index.html js/app.js tutorials/advanced.html tutorials/beginner.html
tutorials/index.html tutorials/intermediate.html""".split()

failures = []
def fail(msg):
    failures.append(msg)
    print("FAIL:", msg)

ATTR = re.compile(r'''(href|src)\s*=\s*(["'])(.*?)\2''', re.I)
HOST = re.compile(r'^https?://(localhost|127\.0\.0\.1)(:8080)?(/.*)?$', re.I)

def norm_target(rel_file, url):
    """Return site-absolute path the link points to, or None if external/other."""
    m = HOST.match(url)
    if m:
        p = m.group(3) or "/"
    elif re.match(r'^[a-z][a-z0-9+.-]*:', url, re.I) or url.startswith("//") or url.startswith("#"):
        return None
    elif url.startswith("/"):
        p = url
    else:
        p = posixpath.join("/" + posixpath.dirname(rel_file), url)
    p = p.split("#")[0].split("?")[0]
    return posixpath.normpath(p)

def normalized(rel_file, text):
    return ATTR.sub(lambda m: '%s="%s"' % (m.group(1).lower(), norm_target(rel_file, m.group(3)) or m.group(3)), text)

def main():
    if not os.path.isdir(MIRROR):
        fail("mirror dir missing"); return
    # 1. completeness + genuine content
    for rel in EXPECTED:
        mp, sp = os.path.join(MIRROR, rel), os.path.join(SRC, rel)
        if not os.path.isfile(mp):
            fail("missing mirrored file %s" % rel); continue
        mb, sb = open(mp, "rb").read(), open(sp, "rb").read()
        if rel.endswith(".html"):
            if normalized(rel, mb.decode("utf-8", "replace")) != normalized(rel, sb.decode("utf-8", "replace")):
                fail("content of %s differs from the served page (beyond link rewriting)" % rel)
        elif mb != sb:
            fail("content of %s differs from the served file" % rel)
    # 2. exclusions (whatever is visible)
    for root, dirs, files in os.walk(MIRROR):
        for d in dirs:
            if d == "admin":
                fail("admin directory present: %s" % os.path.join(root, d))
        for f in files:
            if f.lower().endswith(".zip"):
                fail("zip file present: %s" % os.path.join(root, f))
    # 3. link conversion: links in mirrored HTML must work locally
    for rel in EXPECTED:
        if not rel.endswith(".html"):
            continue
        mp = os.path.join(MIRROR, rel)
        if not os.path.isfile(mp):
            continue
        for m in ATTR.finditer(open(mp, encoding="utf-8", errors="replace").read()):
            url = m.group(3)
            tgt = norm_target(rel, url)
            if tgt is None:
                continue
            local = tgt.lstrip("/")
            if HOST.match(url) or url.startswith("/"):
                # absolute link: allowed only to content that was NOT mirrored
                if local in EXPECTED or os.path.isfile(os.path.join(MIRROR, local)):
                    fail("%s: link %r to mirrored content is not converted to a relative path" % (rel, url))
            else:
                if not os.path.isfile(os.path.join(MIRROR, local)):
                    fail("%s: relative link %r points to a file that is not in the mirror (broken offline)" % (rel, url))
    # 4. manifest
    mf = os.path.join(MIRROR, "mirror_manifest.txt")
    if not os.path.isfile(mf):
        fail("mirror_manifest.txt missing"); return
    entries = {}
    lines = [l for l in open(mf, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
    if not lines:
        fail("manifest empty")
    for line in lines:
        m = re.match(r'^\s*(.+?):\s*(\d+)(\s*bytes?)?\s*$', line, re.I)
        if not m:
            fail("manifest line not in 'filepath: size_in_bytes' format: %r" % line); continue
        p = m.group(1).strip()
        if p.startswith(MIRROR + "/"):
            p = p[len(MIRROR) + 1:]
        p = posixpath.normpath(p.lstrip("./") if p.startswith("./") else p.lstrip("/"))
        if p.startswith("localhost:8080/"):
            p = p[len("localhost:8080/"):]
        entries.setdefault(p, []).append(int(m.group(2)))
        low = p.lower()
        if low.endswith(".zip") or "admin/" in low or "/level6/" in "/" + low:
            fail("manifest lists an excluded file: %r" % line)
    for rel in EXPECTED:
        mp = os.path.join(MIRROR, rel)
        if rel not in entries:
            fail("manifest does not list %s" % rel)
        elif os.path.isfile(mp) and any(s != os.path.getsize(mp) for s in entries[rel]):
            fail("manifest size for %s is %s, actual %d" % (rel, entries[rel], os.path.getsize(mp)))

main()
if failures:
    print("%d check(s) failed" % len(failures)); sys.exit(1)
print("all checks passed"); sys.exit(0)
