"""Documentation links, paper figures and tracking boundaries stay consistent."""

import ast
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mapchain import store


def heading_anchors(text):
    """GitHub-style anchors, including suffixes for repeated headings."""
    counts, anchors = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return anchors


class Layout(unittest.TestCase):
    def test_readme_links_and_images(self):
        for document in (store.ROOT / "README.md", store.OUT / "README.md"):
            text = document.read_text(encoding="utf-8")
            links = re.findall(r"\]\(([^)]+)\)", text)
            self.assertTrue(links, f"No links found in {document}")
            for destination in links:
                with self.subTest(document=document.name, destination=destination):
                    url = urlsplit(destination.strip("<>"))
                    if url.scheme or url.netloc:
                        continue
                    target = document.parent / unquote(url.path) if url.path else document
                    self.assertTrue(target.exists(), f"Missing link target: {target}")
                    if url.fragment:
                        self.assertEqual(target.suffix.lower(), ".md")
                        self.assertIn(
                            unquote(url.fragment),
                            heading_anchors(target.read_text(encoding="utf-8")),
                            f"Missing heading in {target}",
                        )

    def test_figure_correspondence(self):
        # Read the registry without importing the rendering stack or running solvers.
        module = ast.parse((store.ROOT / "mapchain/figures/__init__.py").read_text())
        registered = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "NAMES" for target in node.targets
            )
        )
        self.assertEqual(len(registered), 5)
        self.assertEqual(len(set(registered)), 5)
        pdfs = {path.stem for path in store.FIGURES.glob("*.pdf")}
        references = set()
        for source in (
            store.PAPER / "main.tex",
            *store.PAPER.glob("sections/*.tex"),
            *store.PAPER.glob("appendices/*.tex"),
        ):
            text = re.sub(r"(?m)(?<!\\)%.*$", "", source.read_text(encoding="utf-8"))
            references.update(
                Path(name).stem
                for name in re.findall(
                    r"\\includegraphics(?:orbox)?(?:\[[^\]]*\])?\{figures/([^}]+)\}", text
                )
            )
        self.assertEqual(set(registered), pdfs)
        self.assertEqual(set(registered), references)

    def test_gitignore_tracking_boundaries(self):
        self.assertIsNotNone(shutil.which("git"), "Git is needed to check ignore semantics")
        ignored = (
            ".venv/Scripts/python.exe",
            ".geogram/build/solver.exe",
            ".cache/coacd/hulls.npz",
            ".ruff_cache/entry",
            "mapchain/__pycache__/store.cpython-313.pyc",
            "paper/build/main.aux",
            "paper/build/tikzcache/map-chain.pdf",
        )
        eligible = (
            "models/bunny.stl",
            "results/evaluation/whole_part/bunny_m5.json",
            "results/evaluation/whole_part/bunny_m5.npz",
            "paper/figures/bunny_transport_decomposition.pdf",
            "paper/figures/bunny_transport_decomposition.png",
            "paper/main.pdf",
        )
        # An isolated repository works even when the source is an unpacked archive.
        with tempfile.TemporaryDirectory(prefix="mapchain-ignore-") as directory:
            shutil.copyfile(store.ROOT / ".gitignore", Path(directory) / ".gitignore")
            subprocess.run(["git", "init", "--quiet", directory], check=True, capture_output=True)
            result = subprocess.run(
                ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "-z", "--stdin"],
                cwd=directory,
                input=("\0".join((*ignored, *eligible)) + "\0").encode(),
                capture_output=True,
                check=True,
            )
            self.assertEqual(set(result.stdout.decode().rstrip("\0").split("\0")), set(ignored))


if __name__ == "__main__":
    unittest.main()
