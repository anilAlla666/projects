# Requirements extracted from task instructions

R1. "Determine the topology, the degree of interpenetration, and the average
    internodal distance of the hydrogen-bonded organic frameworks (HOFs)
    provided as ... CIFs in `/app/data`." — an answer must be produced for
    every HOF CIF supplied.

R2. "For every HOF the network is defined exclusively by intermolecular
    hydrogen bonds." (network-construction rule; not independently checkable
    from the JSON output alone, but underlies R5/R6/R7 below)

R3. "A hydrogen bond is identified geometrically as a donor-acceptor (D···A)
    distance of at most 3.5 Å together with a D-H···A angle of at least 120°,
    where the donor (D) and acceptor (A) are electronegative atoms (N, O, F)."
    (geometric definition of the bonds that build the network; not
    independently checkable from the JSON output alone)

R4. "Molecules that do not contribute to the hydrogen-bonded framework must
    be ignored." (network-construction rule; not independently checkable
    from the JSON output alone)

R5. "A node of the topological network is a point at which three or more
    framework molecules are joined together by hydrogen bonding" (molecular
    node or supramolecular/cluster node). => every node/vertex of the
    reported net must have degree (coordination number) >= 3.

R6. "A framework molecule that is hydrogen-bonded to exactly two other
    molecules ... is treated as a single edge between those two nodes and is
    not itself a node." (degree-2 contraction rule feeding into R7; not
    independently checkable from the JSON output alone)

R7. "The average internodal distance is the mean Euclidean distance, in
    nanometers, between pairs of nodes ... that are either directly
    connected ... or connected via a single edge." => the reported distance
    must be expressed in nanometres and must be geometrically plausible for
    the crystal it was measured from (i.e. cannot exceed the span of the
    unit cell the nodes live in).

R8. "Write `/app/solution/output.json` as a JSON object keyed by the file
    stem (filename without the `.cif` extension)." => output file must exist,
    be valid JSON, be a JSON object, and its key set must be exactly the set
    of CIF stems found in `/app/data`.

R9. "Each value is an array of four elements, in this exact order: 1. the
    topology in the RCSR naming scheme (string); 2. the degree of
    interpenetration (integer); 3. the average internodal distance in
    nanometers, rounded to two decimal places, excluding the unit symbol
    (number); 4. the coordination sequence ... (six integers)." => for every
    key:
      - value is a JSON array of exactly length 4
      - element[0] is a non-empty string (RCSR symbol: lower-case
        alphanumeric token, no unit/space/units attached)
      - element[1] is an integer (not a float/string/bool)
      - element[2] is a number, not a string, containing no unit symbol,
        and equal to itself when rounded to 2 decimal places (i.e. it truly
        is given to 2 d.p.)
      - element[3] is a JSON array of exactly 6 elements, each an integer

R10. "The coordination sequence is the translation invariant
    \([c_1,...,c_6]\) in which \(c_k\) is the number of distinct vertices ...
    whose shortest topological path ... is exactly k edges" => for a
    connected, infinite, 3-periodic net every c_k must be a positive integer
    (>=1), and since every vertex/node has degree >= 3 (R5), c_1 (the count
    of nearest neighbours of the reference vertex) must itself be >= 3.

R11. "Report one such sequence per framework" => element[3] has exactly 6
    entries (already covered by R9, listed again because the text repeats
    the constraint explicitly for the coordination sequence).

Notes on what is intentionally NOT checked:
 - The verifier cannot independently re-derive the "true" RCSR topology
   symbol, exact interpenetration degree, or exact coordination sequence
   for each of these seven real crystal structures without redoing the full
   crystallographic analysis (hydrogen-bond perception from the CIF,
   network construction, embedding/interpenetration analysis, and net
   identification against the RCSR database) which is out of scope for a
   deterministic, offline, <10 minute grading script. The checks below
   instead enforce every requirement that IS mechanically checkable from
   the output file together with geometric facts read directly from the
   CIFs (unit cell dimensions), and the internal-consistency constraints
   that follow logically from the task's own node/edge definitions (R5,
   R10). This means a scientifically wrong but well-formed, internally
   consistent, geometrically plausible answer could pass; but no
   requirement stated in the task text is left unchecked, and no
   unstated/guessed numeric thresholds are used (the distance bound is
   derived from each CIF's own unit-cell diagonal, not assumed).
