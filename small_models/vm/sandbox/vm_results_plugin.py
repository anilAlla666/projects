"""Loaded into pytest with -p; appends one line "nodeid<TAB>outcome" to $VM_RESULTS as each test finishes, so a
run killed by the wall-clock limit keeps every result it reached. A test's outcome is its call phase, or its
setup phase when setup did not pass."""
import os


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        path = os.environ.get("VM_RESULTS")
        if path:
            with open(path, "a") as f:
                f.write(f"{report.nodeid}\t{report.outcome}\n")
