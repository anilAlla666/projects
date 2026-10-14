#!/usr/bin/env python3
"""
Tests for Docker container graceful shutdown with signal handling.

These tests verify that the entrypoint.sh has been fixed to properly
forward SIGTERM signals to the Python application.
"""
import os
import re
import sys
import unittest


def read_entrypoint():
    """Read the entrypoint.sh content."""
    entrypoint_path = "/app/entrypoint.sh"
    if not os.path.exists(entrypoint_path):
        raise AssertionError(f"Entrypoint not found at {entrypoint_path}")
    with open(entrypoint_path, "r") as f:
        return f.read()


class TestEntrypoint(unittest.TestCase):
    def test_entrypoint_exists(self):
        """Basic check that entrypoint.sh exists and is executable."""
        entrypoint_path = "/app/entrypoint.sh"
        self.assertTrue(os.path.exists(entrypoint_path), f"Entrypoint not found at {entrypoint_path}")
        self.assertTrue(os.access(entrypoint_path, os.X_OK), "Entrypoint should be executable")

    def test_shebang_and_basic_structure(self):
        """Basic sanity checks."""
        content = read_entrypoint()
        lines = content.split('\n')
        self.assertTrue(lines[0].startswith('#!/bin'), "Entrypoint should start with shebang (#!/bin/bash or #!/bin/sh)")

    def test_uses_exec_or_proper_trap_forward(self):
        """
        Test that the entrypoint uses a correct pattern for signal handling:
        1. Uses 'exec python3 /app/app.py' (or similar) to make Python PID 1, OR
        2. Uses 'trap' to catch SIGTERM, forwards it to child with 'kill -TERM',
           and waits for child to exit cleanly.
        """
        content = read_entrypoint()

        # Pattern 1: exec python (replaces shell, Python becomes PID 1)
        exec_pattern = re.compile(r'\bexec\s+(python3?|/usr/bin/python3?)\s+/app/app\.py\b')
        has_exec = bool(exec_pattern.search(content))

        # Pattern 2: trap + forward SIGTERM to child
        has_trap_term = False
        has_kill_term_child = False
        has_wait_child = False

        # Check for inline trap: trap '...' TERM (including multiline)
        # Use DOTALL to match across newlines
        trap_matches = re.findall(r"trap\s+['\"]([^'\"]*)['\"]\s+(?:TERM|SIGTERM|15)", content, re.DOTALL)
        for trap_body in trap_matches:
            if re.search(r'kill\s+-(?:TERM|15)\s+\$\w+', trap_body):
                has_trap_term = True
                has_kill_term_child = True
            if re.search(r'wait\s+\$\w+', trap_body):
                has_wait_child = True

        # Check for function trap: trap func TERM; func() { ... }
        func_trap_match = re.search(r'trap\s+(\w+)\s+(?:TERM|SIGTERM|15)', content)
        if func_trap_match:
            func_name = func_trap_match.group(1)
            # Escape func_name for regex
            func_pattern = re.escape(func_name) + r'\s*\(\s*\)\s*\{([^}]+)\}'
            func_def = re.search(func_pattern, content)
            if func_def:
                func_body = func_def.group(1)
                if re.search(r'kill\s+-(?:TERM|15)\s+\$\w+', func_body):
                    has_trap_term = True
                    has_kill_term_child = True
                if re.search(r'wait\s+\$\w+', func_body):
                    has_wait_child = True

        has_proper_trap = has_trap_term and has_kill_term_child and has_wait_child

        self.assertTrue(
            has_exec or has_proper_trap,
            "Entrypoint must use either:\n"
            "  1. 'exec python3 /app/app.py' to make Python PID 1, OR\n"
            "  2. A trap that catches SIGTERM, forwards it to the child with "
            "'kill -TERM $child', and waits for the child to exit.\n"
            "Found neither valid pattern in entrypoint.sh"
        )

    def test_no_broken_background_wait_pattern(self):
        """
        Test that the entrypoint doesn't have the broken pattern:
        - Running app in background with &
        - Just calling 'wait' without proper signal forwarding
        """
        content = read_entrypoint()

        # Check for background execution of app.py
        runs_background = bool(re.search(r'python3?\s+/app/app\.py\s+&', content))

        if runs_background:
            # Must have a trap for TERM/SIGTERM (including multiline)
            has_trap = bool(re.search(r'trap\s+.*\s+(?:TERM|SIGTERM|15)', content, re.DOTALL))
            self.assertTrue(
                has_trap,
                "Entrypoint runs app in background (&) but has no trap for SIGTERM. "
                "This causes the container to not respond to docker stop."
            )

    def test_trap_does_not_do_cleanup_in_shell(self):
        """
        Test that if a trap is used, it doesn't duplicate app.py's cleanup logic
        (writing shutdown_complete, etc.) in the shell. The Python app must
        receive SIGTERM and do its own cleanup.
        """
        content = read_entrypoint()

        trap_bodies = []

        # Inline trap: trap '...' SIGTERM (including multiline)
        trap_bodies.extend(re.findall(r"trap\s+['\"]([^'\"]*)['\"]\s+(?:TERM|SIGTERM|15)", content, re.DOTALL))

        # Function trap: trap func SIGTERM; func() { ... }
        func_trap_match = re.search(r'trap\s+(\w+)\s+(?:TERM|SIGTERM|15)', content)
        if func_trap_match:
            func_name = func_trap_match.group(1)
            func_pattern = re.escape(func_name) + r'\s*\(\s*\)\s*\{([^}]+)\}'
            func_def = re.search(func_pattern, content)
            if func_def:
                trap_bodies.append(func_def.group(1))

        for body in trap_bodies:
            # Check if trap body writes shutdown_complete - 
            # this means shell is doing cleanup instead of forwarding to Python
            if 'shutdown_complete' in body:
                # Check for direct file write (>, >>, tee, etc.)
                if re.search(r'(>\s*|>>\s*|tee\s+)/app/data/shutdown_complete', body):
                    self.fail(
                        "Trap handler writes to /app/data/shutdown_complete directly. "
                        "The Python app must receive SIGTERM and write this file itself. "
                        "Forward the signal to the child process instead."
                    )
            if 'END OF DATA' in body:
                if re.search(r'(>\s*|>>\s*|tee\s+)/app/data/output\.txt', body):
                    self.fail(
                        "Trap handler writes 'END OF DATA' to output.txt directly. "
                        "The Python app must receive SIGTERM and do this cleanup itself."
                    )

    def test_no_full_reimplementation(self):
        """
        Test that the entrypoint doesn't reimplement the entire app in bash.
        The task says: "Only modify /app/entrypoint.sh - do not change app.py"
        """
        content = read_entrypoint()

        has_while_loop = bool(re.search(r'while\s+true\s*;', content) or re.search(r'while\s+\[\s+1\s+\]', content))
        has_counter = 'counter' in content
        has_sleep = 'sleep' in content
        writes_data = bool(re.search(r'(echo|printf)\s+.*data_entry', content))

        if has_while_loop and has_counter and has_sleep and writes_data:
            self.fail(
                "Entrypoint appears to reimplement the data processing loop in bash. "
                "The task requires using the existing app.py which has proper signal handlers. "
                "Only modify entrypoint.sh to fix signal forwarding."
            )


if __name__ == '__main__':
    # Run tests and exit with appropriate code
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEntrypoint)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
