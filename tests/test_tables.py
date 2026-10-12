"""The tables in paper/tables are exactly what the saved results produce."""

import json
import re
import unittest

from mapchain import store, tables
from mapchain.evaluation.pieces import SUMMARY


class Tables(unittest.TestCase):
    def test_tables_match_saved_results(self):
        summary = json.loads(SUMMARY.read_text())
        for name, text in (
            ("main_stops", tables.main_stops()),
            ("torus_whole_vs_pieces", tables.torus_whole_vs_pieces()),
            ("sheets_3r_results", tables.sheets_3r_results()),
            ("experiments_geometries", tables.experiments_geometries()),
            ("decomposition_main", tables.decomposition_main(summary)),
        ):
            with self.subTest(name):
                path = store.TABLES / f"{name}.tex"
                self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_every_table_is_included(self):
        sources = "".join(
            p.read_text()
            for p in (*store.PAPER.glob("sections/*.tex"), *store.PAPER.glob("appendices/*.tex"))
        )
        for path in store.TABLES.glob("*.tex"):
            with self.subTest(path.name):
                self.assertIn(f"\\input{{tables/{path.stem}}}", sources)
        for reference in re.findall(r"\\input\{(tables/[^}]+)\}", sources):
            with self.subTest(reference=reference):
                path = store.PAPER / reference
                if not path.suffix:
                    path = path.with_suffix(".tex")
                self.assertTrue(path.is_file(), f"Missing referenced table: {path}")


if __name__ == "__main__":
    unittest.main()
