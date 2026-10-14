# Judgement: Verifier Analysis and Decisions

## Task Requirements Analysis

The task requires creating an offline archive of a documentation website with these specific requirements:

1. **Start HTTP server** on port 8080 serving /var/www/html/ (not directly verifiable from output)
2. **Mirror website** to /home/user/website_mirror/ using wget with:
   - Download all necessary assets (HTML, CSS, JS, images) for offline viewing
   - Convert links to work locally (relative links) via `--convert-links`
   - Rate limit: --wait=0.5 (max 2 requests/second) - not verifiable from output
   - Exclude /admin/ directory
   - Exclude .zip files
   - Max recursion depth: 5 levels
   - Appropriate user-agent string - not verifiable from output
3. **Generate manifest** mirror_manifest.txt with "filepath: size_in_bytes" format

## Source Website Structure (from Dockerfile)

The grading container creates /var/www/html/ with:
- 25 HTML files (excluding admin and depth-6 level6)
- 1 CSS file (css/main.css)
- 1 JS file (js/app.js)
- 2 image files (images/logo.png, images/banner.jpg) - but banner.jpg is NOT linked from any page
- Excluded: /admin/ directory, *.zip files in /downloads/, depth-6 level6 directory

Expected downloaded files: **28 files** (25 HTML + 1 CSS + 1 JS + 1 image)

## Candidate Evaluation

### Candidate A: REJECTED
- **Mirror structure**: 28 files present directly under mirror root ✓
- **Exclusions**: No admin, no zip, no level6 ✓
- **Link conversion**: FAIL - No absolute URLs for excluded content
  - index.html: admin and downloads links remain relative (should be absolute)
  - deep/index.html: level6 link remains relative (should be absolute)
- **Manifest**: FAIL - 56 entries (duplicates with localhost:8080 prefix), lists non-existent banner.jpg
- **Root cause**: Did not use `--convert-links` or used it incorrectly; manifest generation flawed

### Candidate B: ACCEPTED
- **Mirror structure**: 28 files present directly under mirror root ✓
- **Exclusions**: No admin, no zip, no level6, no banner.jpg ✓
- **Link conversion**: PASS - Correct `--convert-links` behavior
  - Downloaded content links: relative (CSS, JS, navigation)
  - Excluded content links: absolute (admin, zip, level6)
- **Manifest**: PASS - 28 entries, correct format, accurate post-conversion sizes
- **Root cause**: Properly executed all wget requirements and manifest generation

### Empty Attempt: REJECTED
- No files present, fails all checks

## Verifier Design Decisions

### Checks Implemented (with basis)

| Check | Requirement Source | Basis |
|-------|-------------------|-------|
| Mirror directory exists with 28 expected files | "mirror directory exists with complete website structure" | task-text |
| No admin directory/files | "Exclude the /admin/ directory" | task-text |
| No .zip files | "Exclude .zip files" | task-text |
| No level6 directory (depth > 5) | "Set a maximum recursion depth of 5 levels" | task-text |
| No banner.jpg (unlinked) | "Download all necessary assets" (only linked assets) | task-text + recomputed |
| Section index files have relative CSS/nav links | "Convert all links in downloaded documents to work locally (relative links)" | task-text |
| Main index has absolute URLs for admin/zip | "Convert all links... to work locally" + wget `--convert-links` behavior | task-text + recomputed |
| Deep index has absolute URL for level6 | "Convert all links... to work locally" + depth limit | task-text + recomputed |
| Manifest exists with correct format | "generate a manifest file called mirror_manifest.txt... format: 'filepath: size_in_bytes'" | task-text |
| Manifest lists exactly 28 expected files | "lists all downloaded files" | task-text |
| Manifest sizes match actual files | "with their sizes" | task-text |

### Checks NOT Implemented (and why)

| Check | Reason |
|-------|--------|
| HTTP server was started | Not observable from output files |
| Rate limit (--wait=0.5) | Not observable from output files |
| User-agent string | Not observable from output files |
| wget command used | Implementation detail; only results matter |

### Contested Values

None. All expected values are settled by:
- **task-text**: Explicit requirements (exclusions, depth limit, manifest format)
- **recomputed**: Expected file list derived from Dockerfile source structure; link conversion behavior derived from wget `--convert-links` documentation applied to known source content

### Unverified Checks

None. All implemented checks are tied to task requirements or recomputed from source data.

## Near Misses Created

Each near miss breaks exactly ONE requirement:

1. **nm1_missing_asset**: Missing images/logo.png (completeness)
2. **nm2_admin_present**: Admin directory present (exclusion)
3. **nm3_zip_present**: .zip file present (exclusion)
4. **nm4_level6_present**: Level6 directory present (depth limit)
5. **nm5_links_not_converted**: Admin link remains relative (link conversion)
6. **nm6_manifest_format**: Manifest has wrong format (manifest)
7. **nm7_manifest_wrong_sizes**: Manifest has incorrect sizes (manifest)
8. **nm8_banner_downloaded**: banner.jpg downloaded (unlinked asset)
9. **nm9_wrong_structure**: Files in localhost:8080 subdirectory (structure)

All near misses are rejected by the verifier.

## Verdict

- **Correct candidates**: B
- **Wrong candidates**: A, empty attempt
- **Verifier changes**: Complete rewrite - removed bogus run.log check, implemented proper validation of all task requirements
