# Grader report: tw-770

Rounds: 1 · cost $0.00 · candidates judged correct: A, C · judged wrong: B, D, E

| basis | checks |
|---|---:|
| task-text | 3 |
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
| Report file exists at /home/user/package_audit_report.txt and contains all 5 required section headers (FILES, DEPENDENCIES, SCRIPTS, CONFFILES, SERVICES) | Create a structured audit report at /home/user/package_audit_report.txt that contains the following sections: 1. FILES SECTION, 2. DEPENDENCIES SECTION, 3. SCRIPTS SECTION, 4. CONFFILES SECTION, 5. SERVICES SECTION | task-text |
| FILES section has category headers (binaries, configuration, libraries, documentation, other), lists critical files (/usr/sbin/nginx, /etc/nginx, nginx.conf, documentation, systemd service, default site), and does not falsely claim '(none)' for configuration or libraries | FILES SECTION: A categorized list of all files installed by the nginx package, grouped by type: Binaries, Configuration files, Libraries, Documentation, Other files | task-text |
| DEPENDENCIES section lists all 7 direct dependencies (libc6, libcrypt1, libpcre2-8-0, libssl3t64, zlib1g, iproute2, nginx-common) with correct version requirements; nginx-common version must be 1.24.0 | DEPENDENCIES SECTION: The complete dependency tree showing: All direct dependencies of the nginx package; Version requirements for each dependency | recomputed |
| SCRIPTS section references postinst scripts, mentions /var/lib/dpkg/info/nginx.postinst, and includes a summary/description of what the scripts do | SCRIPTS SECTION: Information about package scripts: List any postinst (post-installation) scripts associated with nginx; Provide a brief summary of what these scripts do | task-text |
| CONFFILES section does not claim '(none)' or 'no conffiles'; lists at least 3 of 5 key conffiles (nginx.conf, mime.types, sites-available/default, fastcgi_params, proxy_params) | CONFFILES SECTION: A list of configuration files managed by dpkg: These are files marked as 'conffiles' that dpkg tracks for modifications | recomputed |
| SERVICES section mentions nginx.service, includes service unit file path (/lib/systemd/system/nginx.service or /usr/lib/systemd/system/nginx.service), correctly indicates status cannot be determined (systemctl unavailable), and does not falsely claim 'Active: active' | SERVICES SECTION: Information about systemd services: List any systemd service units associated with nginx; Include the service status if possible | recomputed |
