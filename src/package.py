"""Build <team>_submission.zip in the structure required by the challenge."""
import argparse
import zipfile

from config import OUTPUT_DIR, ROOT


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
            z.write(f, f"{code}/src/{f.name}")
        z.write(ROOT / "readme.md", f"{code}/README.md")
        z.write(ROOT / "requirements.txt", f"{code}/requirements.txt")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    print(f"[package] wrote {dest}")


if __name__ == "__main__":
    main()
