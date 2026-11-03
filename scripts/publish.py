"""Copy this repo's figures and glued variables into the paper repo.

    uv run python scripts/publish.py figures   [--paper PATH] [--dry-run] [--only NAME ...]
    uv run python scripts/publish.py variables [--paper PATH] [--dry-run]
    uv run python scripts/publish.py status    [--paper PATH]

The justfile wraps these commands (`just publish-figures`, `just publish-variables`,
`just publish-status`).

`figures` reads figure_manifest.txt and writes each listed file to
<paper>/figures/<basename>, re-encoded by normalize_png. `variables` copies
glued_variables.tex. Neither one checks
the paper: a \\var name with no definition is LaTeX's problem, and manuscript_macros.tex
raises an error for one. `status` reports what both would do, and writes nothing.

Nothing here commits, pushes or pulls. The only git calls are check-ignore and status,
both read-only.

This imports the standard library and Pillow alone, on purpose. `from analysis import ...`
pulls in pyvista, polars, caveclient and sklearn, which costs seconds for a file copy and
fails whenever the scientific environment is broken.
"""

import argparse
import io
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image

# NOTE: as in analysis.io, which this deliberately does not import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "figure_manifest.txt"
VARIABLES = PROJECT_ROOT / "glued_variables.tex"

VAR_RE = re.compile(r"\\var\{([^{}]+)\}")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}")
# NOTE: analysis.utils.encode is the authority for this format.
DEF_RE = re.compile(r"^\\expandafter\\def\\csname (.+)\\endcsname\{(.*)\}$")


def resolve_paper(given: str | None) -> Path:
    """The paper repo: --paper, then $SSA_PATH, then the sibling `ssa` directory."""
    raw = given or os.environ.get("SSA_PATH") or str(PROJECT_ROOT.parent / "ssa")
    paper = Path(raw).expanduser().resolve()
    for name in ("main.tex", "manuscript_macros.tex"):
        if not (paper / name).is_file():
            raise SystemExit(
                f"{paper} does not look like the paper repo ({name} is missing).\n"
                "Pass --paper PATH, set SSA_PATH, or put the repo next to this one."
            )
    return paper


def read_manifest() -> list[Path]:
    """The listed figure paths, relative to the repo root."""
    if not MANIFEST.is_file():
        raise SystemExit(f"no manifest at {MANIFEST}")
    paths: list[Path] = []
    seen: dict[str, int] = {}
    for number, raw in enumerate(MANIFEST.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{MANIFEST.name}:{number}"
        if line.startswith("/") or ".." in Path(line).parts:
            raise SystemExit(f"{where}: path must be relative to the repo root: {line}")
        if not line.endswith(".png"):
            # The paper ignores figures/*.svg and figures/*.pdf, so a non-PNG would
            # copy, show nothing in git status, and look published.
            raise SystemExit(f"{where}: only .png is published: {line}")
        name = Path(line).name
        if name in seen:
            raise SystemExit(f"{where}: {name} is already listed on line {seen[name]}")
        seen[name] = number
        paths.append(Path(line))
    return paths


def normalize_png(source: Path) -> bytes:
    """`source` as an RGB PNG with no ancillary chunks, which pdfTeX copies into the PDF
    as is. An alpha channel or a chunk such as cairosvg's bKGD makes pdfTeX decode and
    re-encode the image, which took 22 of the 24 s of a full manuscript pass. Alpha is
    composited onto white, which loses nothing for the opaque figures this repo makes."""
    # NOTE: the composites are larger than Pillow's decompression-bomb limit.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as image:
        image = image.convert("RGBA")
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    return buffer.getvalue()


def plan_copies(paths: list[Path], paper: Path, only: list[str]) -> list[tuple]:
    """One (action, name, source, destination, data) per listed figure. `data` is the
    normalized PNG, or None for a missing source."""
    actions = []
    for relative in paths:
        name = relative.name
        if only and name not in only:
            continue
        source = PROJECT_ROOT / relative
        destination = paper / "figures" / name
        data = normalize_png(source) if source.is_file() else None
        if data is None:
            action = "missing"
        elif not destination.exists():
            action = "new"
        elif destination.read_bytes() == data:
            action = "same"
        else:
            action = "update"
        actions.append((action, name, source, destination, data))
    return actions


def report_copies(actions: list[tuple]) -> None:
    """Print the plan. The source date is there to catch a long-stale panel."""
    width = max((len(name) for _, name, _, _, _ in actions), default=4)
    source_width = max(
        (len(str(s.relative_to(PROJECT_ROOT))) for _, _, s, _, _ in actions), default=6
    )
    print(f"  {'action':<8}{'figure':<{width + 2}}{'source':<{source_width + 2}}built")
    for action, name, source, _, _ in actions:
        shown = str(source.relative_to(PROJECT_ROOT))
        built = ""
        if source.is_file():
            built = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d")
        print(f"  {action:<8}{name:<{width + 2}}{shown:<{source_width + 2}}{built}")
    counts: dict[str, int] = {}
    for action, _, _, _, _ in actions:
        counts[action] = counts.get(action, 0) + 1
    print("  " + ", ".join(f"{n} {a}" for a, n in sorted(counts.items())))


def git(paper: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(paper), *args], capture_output=True, text=True
    )


def publish_figures(paper: Path, dry_run: bool, only: list[str]) -> None:
    actions = plan_copies(read_manifest(), paper, only)
    if not actions:
        raise SystemExit("nothing selected")
    target = paper / "figures"
    if not target.is_dir():
        raise SystemExit(f"{target} does not exist")
    print(f"paper: {paper}")
    print(f"figures -> {target}" + ("   [DRY RUN -- nothing written]" if dry_run else ""))
    if not dry_run:
        for action, _, _, destination, data in actions:
            if action in ("new", "update"):
                destination.write_bytes(data)
    report_copies(actions)
    copied = [d for a, _, _, d, _ in actions if a in ("new", "update")]
    if copied and not dry_run:
        ignored = [
            d.name
            for d in copied
            if git(paper, "check-ignore", "-q", str(d)).returncode == 0
        ]
        if ignored:
            print("\n  WARNING: the paper's .gitignore hides these, so they will")
            print("  never be committed: " + ", ".join(ignored))
    status = git(paper, "status", "--porcelain", "--", "figures").stdout.strip()
    if status:
        print("\ngit status in the paper repo:")
        print("\n".join("  " + line for line in status.splitlines()))
    print("\nNothing was committed. Review and commit in the paper repo yourself.")
    if any(a == "missing" for a, _, _, _, _ in actions):
        raise SystemExit("\nsome sources are missing; run the scripts that build them")


def publish_variables(paper: Path, dry_run: bool) -> None:
    if not VARIABLES.is_file():
        raise SystemExit(f"no variables file at {VARIABLES}")
    destination = paper / VARIABLES.name
    print(f"paper: {paper}")
    print(f"{VARIABLES.name} -> {destination}" + ("   [DRY RUN]" if dry_run else ""))
    report_variables(paper)
    if not dry_run:
        shutil.copy2(VARIABLES, destination)
    print("\nNothing was committed. Review and commit in the paper repo yourself.")


def used_variables(main_tex: Path) -> tuple[dict, dict]:
    """The \\var names of main.tex, as {name: [line, ...]}, live and comment-only."""
    live: dict[str, list[int]] = {}
    commented: dict[str, list[int]] = {}
    for number, raw in enumerate(main_tex.read_text().splitlines(), start=1):
        cut = re.search(r"(?<!\\)%", raw)
        code = raw[: cut.start()] if cut else raw
        in_code = set(VAR_RE.findall(code))
        for name in in_code:
            live.setdefault(name, []).append(number)
        for name in set(VAR_RE.findall(raw)) - in_code:
            commented.setdefault(name, []).append(number)
    return live, commented


def defined_variables(path: Path) -> dict[str, str]:
    """name -> value, skipping anything that is not a definition."""
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        match = DEF_RE.match(line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def report_variables(paper: Path) -> None:
    live, commented = used_variables(paper / "main.tex")
    new = defined_variables(VARIABLES)
    old = defined_variables(paper / VARIABLES.name)
    missing = sorted(set(live) - set(new))
    if missing:
        print(f"\n  {len(missing)} name(s) used by main.tex are not defined.")
        print("  LaTeX will error on each one and print ??name?? in the PDF:")
        for name in missing:
            lines = ", ".join(f"main.tex:{n}" for n in live[name])
            print(f"    {name:<50}{lines}")
    stale = sorted(set(commented) - set(new) - set(live))
    if stale:
        print(f"  note: {len(stale)} more are used only in commented-out lines.")
    changed = sorted(n for n in live if n in new and n in old and new[n] != old[n])
    if changed:
        width = max(len(n) for n in changed)
        print(f"\n  {len(changed)} value(s) change, of the names main.tex uses:")
        print(f"    {'name':<{width + 2}}{'paper':<22}new")
        for name in changed:
            print(f"    {name:<{width + 2}}{old[name]:<22}{new[name]}")
    added = sorted(n for n in live if n in new and n not in old)
    if added:
        print(f"  {len(added)} name(s) the paper lacks today: {', '.join(added)}")


def show_status(paper: Path) -> None:
    paths = read_manifest()
    print(f"paper: {paper}\n")
    print("figures")
    report_copies(plan_copies(paths, paper, []))

    listed = {p.name for p in paths}
    main_tex = (paper / "main.tex").read_text()
    live_includes, commented_includes = set(), set()
    for raw in main_tex.splitlines():
        cut = re.search(r"(?<!\\)%", raw)
        code = raw[: cut.start()] if cut else raw
        in_code = set(INCLUDE_RE.findall(code))
        live_includes |= in_code
        commented_includes |= set(INCLUDE_RE.findall(raw)) - in_code

    unmanaged = sorted(live_includes - listed)
    if unmanaged:
        print("\n  not in the manifest, so this repo does not publish them:")
        for name in unmanaged:
            print(f"    {name}")
    dead = sorted(listed - live_includes)
    if dead:
        print("\n  in the manifest but no live \\includegraphics uses them:")
        for name in dead:
            print(f"    {name}")
    only_commented = sorted(commented_includes - live_includes)
    if only_commented:
        print("\n  referenced only in commented-out lines: " + ", ".join(only_commented))

    print("\nvariables")
    report_variables(paper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("figures", "variables", "status"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--paper", default=None, help="the paper repo")
        if name != "status":
            sub.add_argument("--dry-run", action="store_true")
        if name == "figures":
            sub.add_argument("--only", nargs="+", default=[], metavar="NAME")
    args = parser.parse_args()
    paper = resolve_paper(args.paper)
    if args.command == "figures":
        publish_figures(paper, args.dry_run, args.only)
    elif args.command == "variables":
        publish_variables(paper, args.dry_run)
    else:
        show_status(paper)


if __name__ == "__main__":
    main()
