# Grader report: tw-1035

Rounds: 1 · cost $0.82 · candidates judged correct: B · judged wrong: A

| basis | checks |
|---|---:|
| task-text | 1 |
| recomputed | 3 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 0 |
| unverified | 1 |

## Needs a human

- **unverified:** wait=0.5 and user-agent not checked (no logs are carried to the grader)

## Every check

| check | requirement | basis |
|---|---|---|
| all 28 files of the wget -l 5 mirror exist (incl. depth-5 deep/index.html) | Download all necessary assets (HTML, CSS, JS, images) for offline viewing; Set a maximum recursion depth of 5 levels | recomputed |
| css/js/png byte-identical to source; HTML identical to source modulo link rewriting | All HTML, CSS, JS, and image files are downloaded correctly | recomputed |
| no absolute link to mirrored content; every relative link resolves to a file in the mirror | Convert all links in downloaded documents to work locally (relative links) | recomputed |
| no admin dir, no .zip file; manifest lists no admin/zip/level6 path | Exclude the /admin/ directory; Exclude .zip files | task-text |
| manifest lines match 'filepath: size' and list all 28 files with actual sizes | generate a manifest file ... lists all downloaded files with their sizes in the format "filepath: size_in_bytes" (one file per line) | format |
| wait=0.5 and user-agent not checked | Respect a reasonable download rate ...; Set an appropriate user-agent string | unverified |
