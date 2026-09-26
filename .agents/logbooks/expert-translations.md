# Expert translation batch (`expert_translation_parser.py`, `moore-web parse-expert-translations`)

Google Sheets export archived as
`../faso-web-docs/expert-translations/mos_Mossi_seed_batch2 (3).pdf`: one
already-paired `fra` → `mos` row per table row, no splitting or alignment.
See `docs/expert-translations.md` for the columns and parser checks.

## 2026-09-26

- **Exported** to `final_data_hf/expert_translations.jsonl`: 352 rows, all
  unique pairs; `qa_check` 324 `ok`, 23 `reviewed`, 5 `flag` (kept as-is).
  Rows use `source_text`/`target_text`, so `build_fr_mos_dataset.py` doesn't
  read them yet: its `_load_jsonl` wants `french`/`moore` and the file isn't
  in its source list.
- **No overlap with facebook/bouquet.** Compared against the
  `fra_Latn-mos_Latn` files only (sentence dev 504 / test 854, paragraph dev
  120 / test 198): 0 exact French or Mooré matches after normalising case,
  quotes and spaces; 0 Bouquet sentences contained in our rows; 0 with char
  ratio ≥ 0.8. Word-overlap "matches" were only function words. Safe for
  training, and Bouquet `fra-mos` stays usable as a clean eval set.
- **Same seed style as Bouquet, different sentences.** Shared `<A:>`/`<B:>`
  dialogue markup and domains (dialogue, narrative, informative,
  instruction-response, casual); IDs look Bouquet-like
  (`manually_created_cmn_Hans_253`) but don't map to Bouquet `uniq_id`s
  (`P001-S1`), so match on text, not id.
