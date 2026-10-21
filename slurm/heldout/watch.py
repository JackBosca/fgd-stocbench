#!/usr/bin/env python3
"""Read-only job picker and log viewer for generation, verification and evaluation."""

import argparse
import datetime
import getpass
import re
import subprocess
from pathlib import Path


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout


def jobs(days):
    since = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    fields = ("id", "name", "state", "elapsed", "log")
    found = {}
    queries = [
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "--expand-patterns",
            "-u",
            getpass.getuser(),
            "-S",
            since,
            "--format=JobID%40,JobName%100,State%40,Elapsed,StdOut%2000",
        ],
        ["squeue", "--array", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T|%M"],
    ]
    for query in queries:
        try:
            for line in command(query).splitlines():
                values = dict(zip(fields, line.strip().split("|")))
                if values.get("id"):
                    found.setdefault(values["id"], {}).update(values)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Cannot query {query[0]}: {error}")
    return list(reversed(list(found.values())))


def show(job, follow=False):
    print(" | ".join(job.get(k, "?") for k in ("id", "name", "state", "elapsed")))
    try:
        details = command(["scontrol", "show", "job", job["id"]])
        for key in ("JobState", "ExitCode", "NodeList", "WorkDir", "StdOut"):
            match = re.search(r"(?:^|\s)" + key + r"=(\S+)", details)
            if match:
                print(f"{key}: {match[1]}")
                if key == "StdOut":
                    job["log"] = match[1]
    except (OSError, subprocess.SubprocessError):
        pass
    log = job.get("log")
    if log and Path(log).is_file():
        subprocess.run(["tail", "-n", "60", *(["-f"] if follow else []), "--", log], check=False)
    else:
        print("Log not available yet.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_id", nargs="?")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    if not 1 <= a.days <= 3650:
        p.error("days must be between 1 and 3650")
    entries = jobs(a.days)
    if a.job_id:
        show(next((j for j in entries if j["id"] == a.job_id), {"id": a.job_id}), a.follow)
        return
    while True:
        for i, job in enumerate(entries):
            print(f"{i:3} {job['id']:16} {job['state']:16} {job['name']}")
        answer = input("Row | /text to filter | r refresh | q quit: ").strip()
        if answer == "q":
            return
        if answer == "r":
            entries = jobs(a.days)
        elif answer.startswith("/"):
            entries = [j for j in jobs(a.days) if answer[1:].lower() in str(j).lower()]
        elif answer.isdigit() and int(answer) < len(entries):
            show(entries[int(answer)], a.follow)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nClosed.")
