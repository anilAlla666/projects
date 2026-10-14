# Verifier Judgement

## Task Requirements Analysis

The task requires creating a comprehensive package audit report for the installed nginx package at `/home/user/package_audit_report.txt` with 5 sections:

1. **FILES SECTION**: Categorized list of all files installed by nginx package, grouped by type (Binaries, Configuration files, Libraries, Documentation, Other files)
2. **DEPENDENCIES SECTION**: Complete dependency tree showing all direct dependencies with version requirements
3. **SCRIPTS SECTION**: Postinst scripts associated with nginx with brief summary
4. **CONFFILES SECTION**: Configuration files managed by dpkg (conffiles)
5. **SERVICES SECTION**: Systemd service units with status

## Ground Truth from System (Ubuntu 24.04)

The nginx package (1.24.0-2ubuntu7.18) is a metapackage that depends on nginx-common. A comprehensive audit must cover both:
- **nginx package files**: /usr/sbin/nginx, docs, man pages
- **nginx-common package files**: /etc/nginx/*, /lib/systemd/system/nginx.service, vim files, apport hooks, etc.

**Direct dependencies with versions** (from `dpkg-query --showformat='${Depends}\n' -W nginx`):
- libc6 (>= 2.34)
- libcrypt1 (>= 1:4.1.0)
- libpcre2-8-0 (>= 10.22)
- libssl3t64 (>= 3.0.0)
- zlib1g (>= 1:1.1.4)
- iproute2 (any version)
- nginx-common (= 1.24.0-2ubuntu7.18)

**Postinst scripts**:
- /var/lib/dpkg/info/nginx.postinst - service upgrade/restart/start, trigger reload
- /var/lib/dpkg/info/nginx-common.postinst - log dir setup, default site, systemd integration

**Conffiles** (from nginx-common, nginx has none):
- 17 files including /etc/nginx/nginx.conf, /etc/nginx/mime.types, /etc/nginx/sites-available/default, etc.

**Systemd service**: /lib/systemd/system/nginx.service (symlink to /usr/lib/systemd/system/nginx.service)
**Service status**: systemctl not available in container

## Candidate Evaluation

### Candidate A: CORRECT ✓
- All 5 sections present and comprehensive
- FILES: Properly categorized, includes both nginx and nginx-common files, no false "none" claims
- DEPENDENCIES: All 7 direct dependencies with correct versions
- SCRIPTS: Both postinst scripts listed with summary and key indicators
- CONFFILES: Complete list of 17 conffiles from nginx-common
- SERVICES: Service unit path, correctly notes systemctl unavailable

### Candidate B: INCORRECT ✗
- FILES: Claims "Configuration files: (none)" and "Libraries: (none)" - false for comprehensive audit
- SCRIPTS: Missing nginx-common.postinst
- CONFFILES: Claims "none reported by dpkg-query for this package" - misleading
- Other sections acceptable

### Candidate C: CORRECT ✓
- All 5 sections present and comprehensive
- FILES: Well categorized with nginx-common files, no false "none"
- DEPENDENCIES: All 7 dependencies with correct versions (nginx-common 1.24.0-2ubuntu7.6 contains 1.24.0)
- SCRIPTS: nginx.postinst with full content and excellent summary
- CONFFILES: Complete with checksums
- SERVICES: Full unit file content, correctly notes status unavailable

### Candidate D: INCORRECT ✗
- FILES: Claims "Configuration files: None" and "Libraries: None" - false
- SCRIPTS: Missing nginx-common.postinst
- CONFFILES: Claims "None" - incorrect
- Other sections acceptable

### Candidate E: INCORRECT ✗
- FILES: Minimal, incomplete categorization
- DEPENDENCIES: Wrong versions (libc6 >= 2.14, nginx-common = 1.18.0)
- SCRIPTS: Just "postinst" with no path or summary
- CONFFILES: Only 1 file listed
- SERVICES: Incorrectly claims "Active: active"

## Current Verifier Issues

The original verifier (`test_outputs.py`) had weak checks:
1. Only checked section headers exist (not content quality)
2. FILES: Only checked for nginx binary OR /etc/nginx mention
3. DEPENDENCIES: Only checked for at least one common dependency pattern
4. CONFFILES: Only checked for /etc/nginx or nginx.conf mention
5. No validation of version accuracy, script summaries, or service status correctness

This caused all 5 candidates to be accepted, including clearly incorrect ones (B, D, E).

## Repaired Verifier Design

New tests validate actual content accuracy against ground truth:
1. **test_report_exists_and_has_sections**: All 5 required sections present
2. **test_files_section_completeness**: Category headers present, critical files listed, no false "(none)" claims
3. **test_dependencies_section_accuracy**: All 7 dependencies named, nginx-common version 1.24.0 verified
4. **test_scripts_section_completeness**: postinst referenced, nginx.postinst path mentioned, summary present
5. **test_conffiles_section_accuracy**: No false "none" claims, at least 3/5 key conffiles listed
6. **test_services_section_accuracy**: nginx.service and path mentioned, status correctly indicates unavailability, rejects "Active: active"

## Near Misses Created

Each near miss breaks exactly one requirement from a correct baseline (Candidate A):

1. **nm1_missing_config_files**: FILES section claims "Configuration files: (none)" 
2. **nm2_wrong_dependency_version**: DEPENDENCIES has nginx-common (= 1.18.0) instead of 1.24.0
3. **nm3_missing_scripts_summary**: SCRIPTS section omits "Brief summary" subsection
4. **nm4_false_conffiles_none**: CONFFILES section claims "(none)"
5. **nm5_false_service_active**: SERVICES claims "Active: active (running)"
6. **nm6_shortcut_fabricated**: Shortcut attempt with fabricated/minimal content across all sections

## Expected Verifier Behavior

- **ACCEPT**: Candidate A, Candidate C
- **REJECT**: Candidate B, Candidate D, Candidate E, all 6 near misses, empty attempt

## Contested Values

None. All expected values are settled by:
- **task-text**: Section requirements explicitly stated
- **recomputed**: Package data from dpkg/apt-cache on this machine
- **format**: Section structure explicitly required

## Unverified Checks

None. All checks are tied to task requirements or recomputed ground truth.
