import pdfplumber
import pandas as pd
import json

PDF_PATH = "Glossaire_des_termes_usuels_du_numerique_et_de_la_poste_en_Moore__valide.pdf"

# Tune line detection: ignore short decorative border lines, snap nearby edges
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 50,  # skip ornamental/border lines shorter than 50pt
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

# Minimum columns a table must have to be considered a glossary table
MIN_COLS = 3


def clean_cell(value: str | None) -> str:
    """Normalize whitespace in a cell value."""
    if value is None:
        return ""
    return " ".join(value.split())


def extract_glossary_tables(pdf_path: str, pages: list[int] | None = None) -> list[dict]:
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        target_pages = pages if pages else range(len(pdf.pages))

        for page_num in target_pages:
            page = pdf.pages[page_num]
            tables = page.extract_tables(TABLE_SETTINGS)

            glossary_tables = [t for t in tables if t and len(t[0]) >= MIN_COLS]

            if not glossary_tables:
                print(f"Page {page_num + 1}: no glossary tables found")
                continue

            for i, table in enumerate(glossary_tables):
                header = [clean_cell(c) for c in table[0]]
                rows = [[clean_cell(c) for c in row] for row in table[1:]]

                print(f"\nPage {page_num + 1}, Table {i + 1} — {len(rows)} entries x {len(header)} cols")
                df = pd.DataFrame(rows, columns=header)
                print(df.to_string(index=False))

                results.append(
                    {
                        "page": page_num + 1,
                        "table_index": i,
                        "columns": header,
                        "rows": rows,
                    }
                )

    return results


def save_results(results: list[dict], out_path: str = "pdfplumber_output.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} table(s) to {out_path}")


if __name__ == "__main__":
    print("=== Testing page 5 ===")
    extract_glossary_tables(PDF_PATH, pages=[4])

    print("\n=== Extracting all pages ===")
    all_results = extract_glossary_tables(PDF_PATH)
    save_results(all_results)
