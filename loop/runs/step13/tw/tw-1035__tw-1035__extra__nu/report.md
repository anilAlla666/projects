# Grader report: tw-1035

Rounds: 1 · cost $0.00 · candidates judged correct: B · judged wrong: A

| basis | checks |
|---|---:|
| task-text | 8 |
| recomputed | 3 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 0 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| Mirror directory exists with all 28 expected files directly under /home/user/website_mirror/ | The mirror directory exists with the complete website structure | task-text |
| No admin directory or files exist in mirror | Exclude the /admin/ directory (do not download any files from it) | task-text |
| No .zip files exist in mirror | Exclude .zip files (do not download any .zip archives) | task-text |
| No level6 directory (depth > 5) exists in mirror | Set a maximum recursion depth of 5 levels | task-text |
| No banner.jpg (unlinked asset) exists in mirror | Download all necessary assets (HTML, CSS, JS, images) for offline viewing | recomputed |
| Section index files (docs, tutorials, api-reference, blog) have relative CSS and navigation links | Convert all links in downloaded documents to work locally (relative links) | task-text |
| Main index.html has absolute URLs for admin and downloads zip links | Convert all links in downloaded documents to work locally (relative links) | recomputed |
| Deep index.html has absolute URL for level6 link | Convert all links in downloaded documents to work locally (relative links) + depth limit of 5 | recomputed |
| mirror_manifest.txt exists with format 'filepath: size_in_bytes' (one per line) | generate a manifest file called mirror_manifest.txt in /home/user/website_mirror/ that lists all downloaded files with their sizes in the format: 'filepath: size_in_bytes' (one file per line) | task-text |
| Manifest lists exactly the 28 expected downloaded files (no more, no less) | lists all downloaded files with their sizes | task-text |
| Manifest file sizes match actual file sizes on disk | with their sizes in the format: 'filepath: size_in_bytes' | task-text |
