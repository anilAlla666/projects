# Requirements extracted from the task

- **R1** — “Determine the topology” and “topology in the RCSR naming scheme (string).”
- **R2** — “the degree of interpenetration (integer).”
- **R3** — “the average internodal distance ... is the mean Euclidean distance, in nanometers” for the specified connected node pairs.
- **R4** — “rounded to two decimal places, excluding the unit symbol (number).”
- **R5** — “the coordination sequence ... given as a JSON array of its first six terms (six integers).”
- **R6** — “For every HOF the network is defined exclusively by intermolecular hydrogen bonds.”
- **R7** — “A hydrogen bond is identified geometrically as a donor–acceptor (D···A) distance of at most 3.5 Å together with a D–H···A angle of at least 120°,” with “donor (D) and acceptor (A) ... N, O, F.”
- **R8** — “Molecules that do not contribute to the hydrogen-bonded framework must be ignored.”
- **R9** — “A node ... [is] a point at which three or more framework molecules are joined together by hydrogen bonding.”
- **R10** — “A node may be of either of the following two kinds, and a single network may contain both kinds at once. If two kinds of node are present, you must consider both.”
- **R11** — “Molecular node: The geometric centroid of a multitopic framework molecule,” whose hydrogen bonds connect it “to three or more neighbouring molecules.”
- **R12** — “Supramolecular (cluster) node: The geometric centroid of a discrete hydrogen-bonded cluster of functional groups in which three or more framework molecules converge and associate through mutual hydrogen bonding of their terminal groups.” “Each molecule that contributes a terminal group to the cluster contributes exactly one edge to it.”
- **R13** — “A framework molecule that is hydrogen-bonded to exactly two other molecules, or that contributes only to two nodes, is treated as a single edge between those two nodes and is not itself a node.”
- **R14** — Average distance considers “pairs of nodes (of either kind) that are either directly connected in the resulting network, or connected via a single edge (a molecule or group that links exactly two nodes).”
- **R15** — “Write `/app/solution/output.json` as a JSON object keyed by the file stem (filename without the `.cif` extension).”
- **R16** — “Each value is an array of four elements, in this exact order”: RCSR topology string, interpenetration integer, distance number, coordination-sequence array.
- **R17** — The coordination sequence is translation invariant, with `c_k` counting distinct periodic-net vertices at shortest-path distance exactly `k`; “Report one such sequence per framework.”
- **R18** — “Do not cheat by using online solutions or hints specific to this task.”

## Verifier-author workflow and packaging requirements

- **R19** — “Write `/verifier_out/requirements.md`. List every requirement the task text states, each as a short quote from the text with an id (R1, R2, ...).”
- **R20** — “Include constraints on *how* the work must be done, not only the final result.”
- **R21** — “Write your tests so that every check names the requirement id it enforces.”
- **R22** — “Do not write any check that is not traceable to a requirement in step 1: a correct solution may legitimately differ from yours in every way the task leaves open.”
- **R23** — “Any numeric threshold must come from the task text or from measuring your own correct solution in step 3, never from a guess about what is ‘plausible’.”
- **R24** — “Solve the task yourself in a scratch copy (for example `/scratch/good`), well enough that it meets every requirement.”
- **R25** — “Then make 2 or 3 flawed variants, each breaking exactly one requirement from step 1, the kind of subtle mistake a solver might make.”
- **R26** — “Run your verifier against your correct solution and each flawed variant, with the files placed at the paths the verifier will see.”
- **R27** — “It must write `1` for the correct solution and `0` for every flawed one.”
- **R28** — “Write the results to `/verifier_out/validation.md`.”
- **R29** — “`Dockerfile`: builds the container your tests run in.” The prescribed base image line is given in the task; “Include `COPY . /tests/`. Do not set an `ENTRYPOINT`.”
- **R30** — “`test.sh`: the grader runs it as `bash /tests/test.sh` inside your container. It must write exactly `1` (the attempt meets every requirement) or `0` (anything else) to `/logs/verifier/reward.txt`, and print which checks failed.”
- **R31** — “Your tests run in a fresh container built from your Dockerfile,” with only `/app/solution/output.json` carried over; “Your tests run with no network access and must finish within 600 seconds.”
- **R32** — “When `/verifier_out/tests/`, `/verifier_out/requirements.md` and `/verifier_out/validation.md` are complete, stop.”
- **R33** — “Accepting a wrong attempt and rejecting a correct one count equally against you.”
- **R34** — “Your verifier will be run on many real attempts. Some are correct; many are subtly wrong (they pass most checks but break one stated requirement).”
- **R35** — “You have 28800 seconds to complete this task.”
- **R36** — “You have 90 minutes.”
