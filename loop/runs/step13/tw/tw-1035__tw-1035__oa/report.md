# Grader report: tw-1035

Rounds: 1 · cost $0.91 · candidates judged correct: B · judged wrong: A

| basis | checks |
|---|---:|
| task-text | 4 |
| recomputed | 5 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 0 |
| unverified | 1 |

## Needs a human

- **unverified:** (not checked) --wait=0.5, user-agent, http server (process-only, so the carried files cannot show it)

## Every check

| check | requirement | basis |
|---|---|---|
| test.sh runs check.py with python3 stdlib only | grader runs bash /tests/test.sh offline | format |
| all 28 pages/assets reachable within depth 5 exist | Download all necessary assets (HTML, CSS, JS, images); complete website structure; max recursion depth 5 | recomputed |
| no level6 page in mirror or manifest | some up to 6 levels, but only download up to 5 | recomputed |
| no admin/ files in mirror or manifest | Exclude the /admin/ directory | task-text |
| no .zip in mirror or manifest | Exclude .zip files | task-text |
| css/js/png byte-identical to site; HTML identical apart from link values | All HTML, CSS, JS, and image files are downloaded correctly | recomputed |
| links to downloaded files are relative and resolve to the same file inside the mirror | Convert all links in downloaded documents to work locally (relative links) | task-text |
| links to non-downloaded files are absolute URLs to the same path (not dangling relative links) | Convert all links ... using wget | recomputed |
| every manifest line is 'path: N' | format: "filepath: size_in_bytes" (one file per line) | format |
| manifest lists every mirrored file with its exact size | lists all downloaded files with their sizes | task-text |
| manifest does not list images/banner.jpg | Mirror the website ... using wget; lists all downloaded files | recomputed |
| (not checked) --wait=0.5, user-agent, http server | Respect a reasonable download rate; Set an appropriate user-agent | unverified |
