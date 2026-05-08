import json
import pandas as pd
from pathlib import Path

XLSX_PATH = Path(__file__).parent / "finaldatasets_2.xlsx"
OUTPUT_DIR = Path(__file__).parent


def parse_sheet(sheet_name):
    raw = pd.read_excel(XLSX_PATH, sheet_name=sheet_name, header=None)

    # Row 0: group headers, Row 1: sub-headers, Row 2+: data
    group_row = raw.iloc[0]
    sub_row   = raw.iloc[1]
    data      = raw.iloc[2:].reset_index(drop=True)

    # Build a flat column map: col_index -> (group, sub_name)
    current_group = None
    col_map = {}
    for i, val in enumerate(group_row):
        if pd.notna(val):
            current_group = str(val).strip()
        sub = str(sub_row[i]).strip() if pd.notna(sub_row[i]) else None
        col_map[i] = (current_group, sub)

    records = []
    for _, row in data.iterrows():
        record = {}

        # Fixed base columns (always present at index 0)
        record["task_id"] = row[0]
        record["original"] = row[1] if pd.notna(row[1]) else ""
        record["entry_point"] = row[2] if pd.notna(row[2]) else ""

        # canonical_solution only in HumanEval (col 3), test_cases shifts
        if sheet_name == "HumanEval":
            record["canonical_solution"] = row[3] if pd.notna(row[3]) else ""
            record["test_cases"] = row[4] if pd.notna(row[4]) else ""
            data_start_col = 5
        else:
            record["test_cases"] = row[3] if pd.notna(row[3]) else ""
            data_start_col = 4

        # Group variant columns by paper
        papers = {}
        for i in range(data_start_col, len(row)):
            group, sub = col_map.get(i, (None, None))
            if group is None or sub is None:
                continue
            # Normalise group name to a clean key
            group_key = group.split(":")[0].strip().lower().replace(" ", "_")
            if group_key not in papers:
                papers[group_key] = {}
            sub_key = (sub.lower()
                          .strip()
                          .replace(" ", "_")
                          .replace("(", "")
                          .replace(")", "")
                          .replace("&", "and")
                          .replace(",", "")
                          .replace("/", "_"))
            papers[group_key][sub_key] = row[i] if pd.notna(row[i]) else ""

        record.update(papers)
        records.append(record)

    return records


for sheet in ["HumanEval", "MBPP"]:
    records = parse_sheet(sheet)
    out_path = OUTPUT_DIR / f"{sheet.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {out_path}")
