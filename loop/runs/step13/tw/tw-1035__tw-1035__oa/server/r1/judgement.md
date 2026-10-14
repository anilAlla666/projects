# Judgement

## State found
- `evidence.md` contains no runs (probe crashed on a missing grader config), so I ran everything myself.
- The original `test.sh` installs curl and uv via apt and curl and then adds pytest from the network. The grading container has no network and no pytest,
  so it would write 0 for **every** attempt, including correct ones. I replaced it with a pure-python3 stdlib checker (`check.py`).
- Only candidates A and B exist in `/candidates/`.

## Reference run (recomputed, procedure named by the task)
I served /var/www/html on :8080 and ran `wget -r -l 5 -p -k -E -nH -X /admin -R '*.zip' --wait=0.5 -U ...`.
Result: exactly the 28 files in the carried-path list (level6 is at depth 6 and is not fetched; banner.jpg is not linked from any page and is not fetched).
Variants checked: `-m -l5 -k -p -E` (no -nH) gives the same bytes under localhost:8080/. `--reject-regex` instead of -X/-R also gives the same bytes.
In every `--convert-links` run, the HTML is the source page except that links to files that were **not** downloaded
(admin/index.html, downloads/archive.zip, deep/level6/index.html) become absolute `http://localhost:8080/...` URLs.
This is what `man wget` documents for -k: "links to files that have not been downloaded by Wget will be changed to include host name and absolute path".
`--convert-file-only` instead produces broken `../../index.html` links (it fails, which is correct).

## Candidates
- **B: correct.** All 28 files are byte-identical to my reference wget -k run. Every manifest entry is `./path: size` with the correct sizes.
- **A: wrong.** Its HTML is byte-identical to the unmodified source in /var/www/html, so no link conversion was done. The links to admin/,
  downloads/archive.zip and level6/ stay relative and point at files that do not exist locally. Its manifest also lists `images/banner.jpg` (and
  `localhost:8080/images/banner.jpg`). No page links to that file, so a recursive wget crawl cannot have downloaded it. This shows the
  files were copied from /var/www/html rather than mirrored. Its `localhost:8080/index.html` is listed at 571 bytes, the unconverted
  source size; the converted size is 615. My checker rejects A on the dangling-link check (3 links) and on the banner.jpg manifest check.
  The original verifier would have accepted A (apart from the network problem).
- Empty attempt: rejected (missing files).

## Trace: requirement → check
| Task requirement | Check in check.py |
|---|---|
| "mirror directory exists with the complete website structure"; "All HTML, CSS, JS, and image files are downloaded" | all 28 expected files exist (step 1) |
| "Set a maximum recursion depth of 5" / "some up to 6 levels, but only download up to 5" | deep/index.html (depth 5) must exist; no level6 path in the mirror or the manifest |
| "downloaded correctly" | css/js/png byte-identical to the site. HTML equals the served page once link attribute values are blanked, and the number of links is the same |
| "Convert all links ... to work locally (relative links)" | every link to a downloaded file is relative and resolves to the same file (and does not escape the mirror root). Links to non-downloaded files must be absolute URLs to the same path, as wget -k makes them, not dangling local links |
| "Exclude the /admin/ directory" | no `admin` dir or file in the mirror; manifest does not list admin |
| "Exclude .zip files" | no *.zip in the mirror; manifest does not list .zip |
| manifest "lists all downloaded files with their sizes in the format: 'filepath: size_in_bytes' (one file per line)" | every non-empty line matches `path: N`. All 28 files are listed with their exact sizes (`./`, bare or absolute path prefixes all accepted). Extra entries are allowed (e.g. the manifest itself) except excluded files and banner.jpg, which cannot have been downloaded |
| --wait=0.5, user-agent, "start a local HTTP server" | process requirements. Nothing in the carried output shows them, so they are not checkable (unverified) |

## Original checks → disposition
- test_mirror_completeness (>=15 html, any css/js/image): replaced by the exact 28-file list from the reference run (stricter, recomputed).
- test_link_conversion (absolute links allowed in up to 1 of 4 section pages; only CSS or JS needed to be relative): the tolerance
  is not in the task ("Convert all links"), so it is replaced by a per-link check.
- test_exclusion_rules: kept and extended to the manifest.
- test_manifest_generation (80% lines contain ':' or 'path N', >=10 lines): loosened against the task's format, so it is replaced by an
  exact format check plus all files listed with correct sizes.

## Near misses (all rejected; B accepted)
nm_unconverted_link, nm_root_absolute_link, nm_manifest_wrong_size, nm_manifest_missing_entry, nm_manifest_format,
nm_depth6_downloaded, nm_depth4_only, nm_zip_downloaded, nm_copy_shortcut (source copied plus a correct manifest).

## Contested
None. There are no numeric or label values whose reading the task leaves open. The expected file set and the link forms come from
running the wget procedure the task names, and every flag variant I tried gives the same bytes.
