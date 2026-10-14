# Grader report: tw-1035

Rounds: 1 · cost $0.81 · candidates judged correct: B · judged wrong: A

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 2 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 0 |
| contested | 0 |
| unverified | 1 |

## Needs a human

- **unverified:** wait/user-agent/server port (not observable from carried outputs; no check)

## Every check

| check | requirement | basis |
|---|---|---|
| all 28 files reachable within depth 5 (excluding admin/zip) exist; assets byte-identical to source; HTML identical modulo link rewriting | Download all necessary assets (HTML, CSS, JS, images) for offline viewing / Set a maximum recursion depth of 5 levels / All HTML, CSS, JS, and image files are downloaded correctly | recomputed |
| links to mirrored files are relative; every relative link resolves to an existing local file | Convert all links in downloaded documents to work locally (relative links) | task-text |
| no admin/, *.zip or level6 path on disk | Exclude the /admin/ directory ... Exclude .zip files ... only download up to 5 | task-text |
| manifest lines match 'filepath: size'; every mirrored file listed with its exact byte size; no excluded paths listed | generate a manifest file called mirror_manifest.txt ... in the format: "filepath: size_in_bytes" (one file per line) | recomputed |
| wait/user-agent/server port | --wait=0.5, user-agent, port 8080 | unverified |
