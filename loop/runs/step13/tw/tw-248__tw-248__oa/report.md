# Grader report: tw-248

Rounds: 1 · cost $0.70 · candidates judged correct: C · judged wrong: A, B, D, E

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 5 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| report file exists and is non-empty | The report should be saved to a file called network_driver_report.txt in the working directory. | format |
| loopback 'lo' is reported | The report must include for each network interface ... The system has various network interfaces including ... loopback | task-text |
| all six items covered (driver, hardware description, module path, dependencies, version, parameters) | The report must include for each network interface: 1..6 | format |
| interface section with a named driver states path, dependencies, parameters | 3. The driver module's full filesystem path 4. Module dependencies 6. Module parameters currently in use (if any) | format |
| interface section with a named driver gives a concrete version (contains a digit) | 5. Driver version information | task-text |
| no placeholder/TODO text | gather this information programmatically | format |
| all kernel releases stated in the report agree | use appropriate system introspection tools to gather this information programmatically | format |
