# Judgement

## Setup
Reproduced the task here: served /var/www/html on :8080 and ran
`wget -r -l 5 -p -k -E -nH --wait=0.5 -X /admin -R zip -U "Mozilla/5.0 (OfflineMirror)" http://localhost:8080/`
(GNU Wget 1.21.4). Result: 28 files (24 HTML + css/main.css, js/app.js, images/logo.png), exactly the
carried-path list. `diff -r` against candidate B: identical apart from the manifest. images/banner.jpg is not linked
from any page, so it is not part of the mirror; docs/.../deep/index.html is link depth 5 and is included; level6 is depth 6 and excluded.
wget -k (man wget, --convert-links) rewrites links to files that were not downloaded (admin, zip, level6) as
absolute `http://localhost:8080/...` URLs, and leaves links to downloaded files relative.

## Candidates
- **A: wrong** (the original verifier accepted it; the repaired one rejects it). Every carried HTML file in A is byte-identical to the source in
  /var/www/html (`diff -r` shows only the excluded files missing). No link was converted. index.html keeps
  `href="admin/index.html"` and `href="downloads/archive.zip"`, and deep/index.html keeps `href="level6/index.html"`, all relative
  links to files that are not in the mirror, so they do not work locally. The task says "Convert all links in downloaded documents to work locally".
  wget --convert-links would have rewritten these links, so A never applied link conversion. (A's manifest also lists a duplicate
  `localhost:8080/` tree and images/banner.jpg, which no page links to. That suggests a copy of the web root, not a wget mirror, but the verdict does not depend on it.)
  Rejected by the new check `test_links_work_locally`: "relative link admin/index.html does not resolve to a local file".
- **B: correct.** Byte-identical to my own wget run. The manifest lists every file with its correct size and nothing excluded.
- **empty attempt: wrong** (no files).

## Trace: requirement -> check
| Task requirement | Check |
|---|---|
| "Download all necessary assets (HTML, CSS, JS, images)", "complete website structure", "files are downloaded correctly" | test_expected_files_present_and_correct: all 28 reachable files exist; CSS/JS/PNG are byte-identical to the source; HTML is identical after every href/src is replaced by the site path it resolves to |
| "Set a maximum recursion depth of 5 levels" / "some up to 6 levels, but only download up to 5" | depth-5 page deep/index.html is required (above); level6 is rejected if present on disk or in the manifest (test_exclusions, test_manifest) |
| "Convert all links ... to work locally (relative links)" | test_links_work_locally: a link to mirrored content must not be absolute; every relative link must resolve to an existing file in the mirror |
| "Exclude the /admin/ directory", "Exclude .zip files" | test_exclusions (on disk) and test_manifest (the manifest must not list admin/, *.zip or level6). The excluded paths themselves are not carried to the grader, so the manifest is the only record of them |
| manifest "filepath: size_in_bytes (one file per line)" | test_manifest: each non-blank line matches `path: digits`; every mirrored file is listed with its exact size (`./`, bare and absolute path prefixes are all accepted). Listing the manifest itself is optional |
| --wait=0.5, user-agent, port 8080 server | not observable in the output; **unverified** (no check) |

## Trace: check -> requirement (old checks)
- The old test_mirror_completeness asked for ">=15 HTML" and any css/js/image. It is replaced by the exact reachable set, which my wget run recomputed.
- The old test_link_conversion only looked for absolute links in 4 files, with a tolerance of 1 file. It is tightened: the task says "all links".
- The old exclusion test is kept and extended to the manifest.
- The old manifest test accepted 80% of lines with any ':' and at least 10 lines. It is replaced by a strict format plus correct sizes ("size_in_bytes" is recomputed with stat).
- test.sh: dropped apt/uv/pytest bootstrapping, because the grading container has no network. The tests now run as a plain python3 script. A pristine copy of the site is kept in tests/site/ for content comparison.

## Contested
None. Every expected value is either the source files themselves or a value recomputed with wget/stat as the task specifies.

## Near misses (all rejected; B and a variant of B with absolute manifest paths accepted)
nm1_depth4_only, nm2_manifest_wrong_size, nm3_unconverted_css_link, nm4_manifest_format_bytes, nm5_admin_downloaded,
nm6_corrupt_css, nm7_shortcut_placeholders (fabricated placeholder pages + matching manifest), nm8_manifest_missing_entry.
