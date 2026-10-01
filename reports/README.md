# Evidence archive

`historical/` contains retained experiment reports. Most are copied verbatim. Five example-bearing reports are truncated before the first record-level example section: EDA, EDA2, normalization, error analysis, and V9 remaining misses. Each excerpt has a visible redaction note.

`manifests/source_archive.json` lists original and retained file hashes, destinations, and transformations. `artifact_inventory.json` records hashes and sizes of models and final outputs that were inspected but excluded from Git and removed from the old workspace. The submission file checksums were compared with the export report before removal.

Absolute workspace prefixes were removed from configuration JSON. Relative paths identify historical artifacts; they do not imply that those files remain available. The learned alias map is preserved with historical source, and its earlier split differs from the later model split.

Use [results.md](../docs/results.md) for interpretation. In particular, `selection_rules.json` is the only retained record of the later 0.931282 development score; the full evaluation sweep was already deleted. The approximately 0.88 public leaderboard score was reported by the owner in conversation and has no retained portal export.
