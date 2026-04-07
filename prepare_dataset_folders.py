import argparse
import csv
import shutil
from pathlib import Path


CLASS_NAMES = {"0": "Down", "1": "Neutral", "2": "Up"}
DEFAULT_CSV = r"D:\full-stack2\data\labels\labeled_dataset.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default="organized_dataset")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating a manifest only.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "label", "ticker", "date", "source_path", "organized_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            split = "validation" if row["split"] == "val" else row["split"]
            label = CLASS_NAMES.get(row["label_id"], row.get("label", "Unknown"))
            source = Path(row["image_path"])
            target = output_dir / split / label / f"{row['ticker']}_{row['date']}.png"

            if args.copy and source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

            writer.writerow({
                "split": split,
                "label": label,
                "ticker": row["ticker"],
                "date": row["date"],
                "source_path": str(source),
                "organized_path": str(target),
            })

    print(f"Manifest saved to: {manifest_path.resolve()}")
    if args.copy:
        print(f"Images copied under: {output_dir.resolve()}")
    else:
        print("Use --copy if you also want physical train/validation/test folders.")


if __name__ == "__main__":
    main()
