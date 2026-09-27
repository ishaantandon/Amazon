"""Build <team>_submission.zip in the structure required by the challenge."""
import argparse
import zipfile

from config import OUTPUT_DIR, ROOT

# The final pipeline's scripts live in tools/ in this repo; the package puts all source under src/.
# They locate src/ relative to their own file and import each other by module name, so they run there unchanged.
PIPELINE_TOOLS = ["faithful_sim.py", "ce_pilot.py", "ce_v5.py", "ce_v8.py", "ce_v9.py", "dense_probe.py",
                  "dense_rescue.py"]
EXCLUDE_SRC = {"stage2_next.py"}  # abandoned experiment, never validated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="aid")
    a = ap.parse_args()
    dest = ROOT / f"{a.team}_submission.zip"
    code = "code/business_entity_resolution"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(OUTPUT_DIR / f, f"output/{f}")
        for f in sorted((ROOT / "src").glob("*.py")):
            if f.name not in EXCLUDE_SRC:
                z.write(f, f"{code}/src/{f.name}")
        for name in PIPELINE_TOOLS:
            z.write(ROOT / "tools" / name, f"{code}/src/{name}")
        z.write(ROOT / "readme.md", f"{code}/README.md")
        z.write(ROOT / "requirements.txt", f"{code}/requirements.txt")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    print(f"[package] wrote {dest}")


if __name__ == "__main__":
    main()
