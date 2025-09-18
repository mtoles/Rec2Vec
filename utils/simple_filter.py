#!/usr/bin/env python3
"""
Simple filter to extract only user-defined functions from cProfile output.
"""

import re
import sys
from pathlib import Path


def extract_user_functions(profile_file, output_file=None):
    """
    Extract only user-defined functions from the project files.
    """
    project_files = {"main.py", "test.py", "utils/retry.py", "utils/sat.py"}

    # Pattern to match function calls from project files
    project_pattern = re.compile(
        r"^\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+("
        + "|".join(re.escape(f) for f in project_files)
        + r"):(\d+)\(([^)]+)\)$"
    )

    user_functions = []

    with open(profile_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            match = project_pattern.match(line)
            if match:
                (
                    ncalls,
                    tottime,
                    percall,
                    cumtime,
                    percall2,
                    filename,
                    lineno,
                    funcname,
                ) = match.groups()
                user_functions.append(
                    {
                        "ncalls": ncalls,
                        "tottime": float(tottime),
                        "percall": float(percall),
                        "cumtime": float(cumtime),
                        "percall2": float(percall2),
                        "filename": filename,
                        "lineno": int(lineno),
                        "funcname": funcname,
                    }
                )

    # Sort by cumulative time (most time-consuming first)
    user_functions.sort(key=lambda x: x["cumtime"], reverse=True)

    # Format output
    output_lines = [
        "=" * 60,
        "USER-DEFINED FUNCTIONS PROFILE",
        "=" * 60,
        f"Total user-defined functions: {len(user_functions)}",
        "",
        f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} {'filename:line(function)':<40}",
        "-" * 80,
    ]

    for func in user_functions:
        output_lines.append(
            f"{func['ncalls']:>8} {func['tottime']:>8.3f} {func['percall']:>8.3f} "
            f"{func['cumtime']:>8.3f} {func['percall2']:>8.3f} "
            f"{func['filename']}:{func['lineno']}({func['funcname']})"
        )

    output_text = "\n".join(output_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"User-defined functions profile saved to: {output_file}")
    else:
        print(output_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_filter.py <profile.txt> [output.txt]")
        print("If no output file is specified, results will be printed to stdout")
        sys.exit(1)

    profile_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(profile_file).exists():
        print(f"Error: Profile file '{profile_file}' not found")
        sys.exit(1)

    extract_user_functions(profile_file, output_file)


if __name__ == "__main__":
    main()
