#!/usr/bin/env python3
"""
Extract and display top user-defined functions by various metrics.
"""

import re
import sys
from pathlib import Path


def analyze_user_functions(profile_file):
    """
    Analyze user-defined functions and show top performers by different metrics.
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
                        "full_name": f"{filename}:{lineno}({funcname})",
                    }
                )

    if not user_functions:
        print("No user-defined functions found in profile.")
        return

    print("=" * 80)
    print("USER-DEFINED FUNCTIONS ANALYSIS")
    print("=" * 80)
    print(f"Total user-defined functions: {len(user_functions)}")
    print()

    # Top functions by cumulative time
    print("TOP 10 FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 80)
    print(
        f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} {'function':<40}"
    )
    print("-" * 80)
    top_cumtime = sorted(user_functions, key=lambda x: x["cumtime"], reverse=True)[:10]
    for func in top_cumtime:
        print(
            f"{func['ncalls']:>8} {func['tottime']:>8.3f} {func['percall']:>8.3f} "
            f"{func['cumtime']:>8.3f} {func['percall2']:>8.3f} {func['full_name']:<40}"
        )

    print()

    # Top functions by total time
    print("TOP 10 FUNCTIONS BY TOTAL TIME:")
    print("-" * 80)
    print(
        f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} {'function':<40}"
    )
    print("-" * 80)
    top_tottime = sorted(user_functions, key=lambda x: x["tottime"], reverse=True)[:10]
    for func in top_tottime:
        print(
            f"{func['ncalls']:>8} {func['tottime']:>8.3f} {func['percall']:>8.3f} "
            f"{func['cumtime']:>8.3f} {func['percall2']:>8.3f} {func['full_name']:<40}"
        )

    print()

    # Most called functions
    print("MOST CALLED FUNCTIONS:")
    print("-" * 80)
    print(
        f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} {'function':<40}"
    )
    print("-" * 80)
    most_called = sorted(
        user_functions, key=lambda x: int(x["ncalls"].split("/")[0]), reverse=True
    )[:10]
    for func in most_called:
        print(
            f"{func['ncalls']:>8} {func['tottime']:>8.3f} {func['percall']:>8.3f} "
            f"{func['cumtime']:>8.3f} {func['percall2']:>8.3f} {func['full_name']:<40}"
        )

    print()

    # Functions with significant time (> 0.001 seconds)
    print("FUNCTIONS WITH SIGNIFICANT TIME (> 0.001s):")
    print("-" * 80)
    print(
        f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} {'function':<40}"
    )
    print("-" * 80)
    significant = [
        f for f in user_functions if f["tottime"] > 0.001 or f["cumtime"] > 0.001
    ]
    significant.sort(key=lambda x: x["cumtime"], reverse=True)
    for func in significant:
        print(
            f"{func['ncalls']:>8} {func['tottime']:>8.3f} {func['percall']:>8.3f} "
            f"{func['cumtime']:>8.3f} {func['percall2']:>8.3f} {func['full_name']:<40}"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python top_functions.py <profile.txt>")
        sys.exit(1)

    profile_file = sys.argv[1]

    if not Path(profile_file).exists():
        print(f"Error: Profile file '{profile_file}' not found")
        sys.exit(1)

    analyze_user_functions(profile_file)


if __name__ == "__main__":
    main()
