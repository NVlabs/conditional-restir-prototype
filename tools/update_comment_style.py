#!/usr/bin/env python

"""
A utility for converting the old comment style to the new comment style.
"""

import sys
import os
import re
import argparse


DEFAULT_EXTENSIONS = "h,cpp,slang,slangh"


def list_files(files, recursive=False, extensions=[]):
    out = []
    for file in files:
        file = os.path.normpath(file)
        if recursive and os.path.isdir(file):
            for dirpath, dnames, fnames in os.walk(file):
                fpaths = [os.path.join(dirpath, fname) for fname in fnames]
                for f in fpaths:
                    ext = os.path.splitext(f)[1][1:]
                    if ext in extensions:
                        out.append(f)
        else:
            out.append(file)
    return out


def prefix_length(s):
    m = re.match("(\s+)", s)
    if m:
        return len(m.group(0))
    return 0


def format_comment(comment):
    indent = prefix_length(comment) * " "
    lines = comment.splitlines()
    new_lines = [indent + "/**"]

    is_new_style = True

    for index, line in enumerate(lines):
        line = line.replace("/**", "")
        line = line.replace("*/", "")
        line = line.replace("\\brief", "@brief")
        line = line.replace("\\param", "@param")
        line = line.replace("\\return", "@return")
        line = line.strip()
        if line == "" and (index == 0 or index == len(lines) - 1):
            continue
        if not line.startswith("*"):
            is_new_style = False
        line = indent + " * " + line
        new_lines.append(line)

    new_lines.append(indent + " */")

    return comment if is_new_style else "\n".join(new_lines)


def fix_comments(text):
    r = re.compile(r"^[ \t]*\/\*\*.+?\*\/$", re.DOTALL | re.MULTILINE)

    matches = list(r.finditer(text))
    for m in reversed(matches):
        comment = m.group()
        start = m.start()
        end = m.end()

        if "***" in comment:
            continue

        new_comment = format_comment(comment)
        text = text[0:start] + new_comment + text[end:]

    return text


def run(args):
    files = list_files(
        args.files, recursive=args.recursive, extensions=args.extensions.split(",")
    )

    for file in files:
        text = open(file, "r").read()
        print(f"Checking file '{file}' ...")
        original = text
        text = fix_comments(text)
        if not args.dry_run and text != original:
            print(f"Writing file '{file}'")
            open(file, "w").write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extensions",
        help="comma separated list of file extensions (default: {})".format(
            DEFAULT_EXTENSIONS
        ),
        default=DEFAULT_EXTENSIONS,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="run recursively over directories",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without writing files",
    )
    parser.add_argument("files", metavar="file", nargs="+")

    args = parser.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
