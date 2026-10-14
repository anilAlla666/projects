# Judgement

## Reference run (recomputed)
On this machine I served /var/www/html with `python3 -m http.server 8080` and ran
`wget --mirror -l 5 --page-requisites --convert-links --adjust-extension --no-parent --wait=0.5 -X /admin -R zip -U ... -nH`.
Result: exactly the 28 content files in the carried-over path list (26 HTML, css/main.css, js/app.js, images/logo.png).
Pages are byte-identical to the source except where they link to content that was not downloaded: wget -k (manual:
"links to files that have not been downloaded by Wget will be changed to include host name and absolute URL")
rewrote `admin/index.html`, `downloads/archive.zip` (index.html) and `level6/index.html` (deep/index.html) to
`http://localhost:8080/...`. Variants (`-r -l 5 -k`, with/without `-p`, `-E`) give the same links. Candidate B is
byte-identical to my run.

## Candidates
- **A: WRONG** (the old verifier accepted it; the new one rejects it). Every HTML file is byte-identical to the
  /var/www/html source, so no link conversion happened. index.html still has `href="admin/index.html"` and
  `href="downloads/archive.zip"`, and deep/index.html has `href="level6/index.html"`. All three are relative links to
  files that are not in the mirror, so they are broken offline. That breaks "Convert all links in downloaded documents
  to work locally", and it is not what wget --convert-links produces. Its manifest also lists `images/banner.jpg`,
  which no page links to, so wget could not have downloaded it. This shows the tree was copied from /var/www/html
  rather than mirrored. Rejected by: "relative link ... points to a file that is not in the mirror".
- **B: CORRECT.** It is identical to my reference wget run. The manifest lists the 28 files with correct sizes.
- **Empty attempt:** rejected (files missing, no manifest).

## Trace: requirement -> check
- "Download all necessary assets (HTML, CSS, JS, images)", "mirror directory exists with the complete website
  structure" -> all 28 expected files must exist (set recomputed by wget -l 5).
- "All HTML, CSS, JS, and image files are downloaded correctly" -> css/js/png must be byte-identical to the source.
  HTML must equal the source after href/src values are normalised to site paths, so link rewriting is allowed and
  fabricated content is not.
- "Convert all links ... to work locally (relative links)" -> (a) no absolute (http://localhost or /-rooted) link
  to mirrored content; (b) every relative link must resolve to a file in the mirror.
- "Exclude /admin/", "Exclude .zip" -> no admin dir and no .zip file in the visible mirror. The manifest must not
  list admin, .zip or level6 paths. A relative link to admin/zip in index.html also fails (b), which catches
  download-then-delete.
- "max recursion depth of 5" -> deep/index.html (depth 5) is required. A relative level6 link fails (b), which
  catches depth 6. level6 files themselves are not carried over to the grader, so only these indirect checks are
  possible.
- "manifest ... lists all downloaded files with their sizes in the format "filepath: size_in_bytes" (one file per
  line)" -> every non-empty line must match `path: <int>` (an optional "bytes" suffix is tolerated). The path may be
  relative, ./ or absolute. All 28 files must be listed with sizes equal to the actual file sizes.
- `--wait=0.5` and user-agent -> **unverified**: no logs are carried to the grader and the output does not depend on
  them.

## Checks removed or changed from the original
- Ran through uv/pytest, which needs network -> replaced by a plain python3 script.
- "<=1 file with absolute links tolerated" -> removed. The task says *all* links.
- ">=15 HTML files" -> replaced by the exact recomputed file set.
- Manifest check "80% of lines contain ':'" and ">=10 lines" -> replaced by a strict format check plus
  completeness and correct sizes.
- Main index "relative css or js" -> covered by the general link checks.

## Near misses (all rejected by the new verifier)
nm_depth6_link, nm_admin_downloaded, nm_zip_downloaded, nm_depth4_only, nm_unconverted_css,
nm_manifest_wrong_size, nm_manifest_format, nm_manifest_incomplete, nm_fabricated_page, nm_shortcut_copy (a plain
copy of the source tree, i.e. a shortcut).

## Contested
None. The expected file set and link forms were recomputed with wget, the tool the task names.
