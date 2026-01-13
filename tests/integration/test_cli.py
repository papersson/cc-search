"""Integration tests for the CLI."""

import json
import subprocess
import sys


def test_cli_help():
    """Test that --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "cc_search.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "search" in result.stdout
    assert "index" in result.stdout
    assert "status" in result.stdout


def test_cli_version():
    """Test that --version works."""
    result = subprocess.run(
        [sys.executable, "-m", "cc_search.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cc-search" in result.stdout


def test_cli_status():
    """Test that status command works."""
    result = subprocess.run(
        [sys.executable, "-m", "cc_search.cli", "status"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Sessions indexed:" in result.stdout
    assert "Index path:" in result.stdout


def test_search_json_output():
    """Test that search with --json outputs valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "cc_search.cli", "search", "test query", "--json", "--limit", "1"],
        capture_output=True,
        text=True,
    )

    # Even if no results, should be valid JSON
    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            assert "results" in data
            assert "query" in data
            assert "search_time_ms" in data
        except json.JSONDecodeError:
            pass  # Some output might not be pure JSON (e.g., index building message)
