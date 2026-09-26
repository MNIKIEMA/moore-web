"""moore-web CLI — bilingual French/Mooré corpus pipeline.

Pipeline stages
---------------
    parse      → flatten → align
    parse-flat →          align
    e2e (all stages in one command)

Sources
-------
    sida   Bilingual SIDA book (single PDF, columns interleaved)
    kade   Kadé facilitator manuals (two separate PDF/TXT files)
    news   Raamde news corpus (JSON with ``text_units`` lists)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Optional, cast

import msgspec
import typer

app = typer.Typer(
    name="moore-web",
    help="Bilingual French/Mooré corpus pipeline.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Source(str, Enum):
    sida = "sida"
    kade = "kade"
    news = "news"
    simple = "simple"
    one_column_dict = "one-column-dict"
    conseils = "conseils"
    digital = "digital"
    udhr = "udhr"
    abc_coepouses = "abc-coepouses"
    moore_tales = "moore-tales"


class KadeLang(str, Enum):
    french = "french"
    moore = "moore"


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"moore-web {version('moore-web')}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Bilingual French/Mooré corpus pipeline: parse → flatten → align."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_output(input_path: Path, suffix: str) -> Path:
    """Derive a default output path from the input stem."""
    return input_path.with_name(input_path.stem + suffix)


def _err(msg: str) -> None:
    typer.echo(f"Error: {msg}", err=True)


@app.command("prepare-new-year-message")
def prepare_new_year_message(
    collection_dir: Annotated[
        Path,
        typer.Option("--collection-dir", exists=True, file_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Prepare a manifest-backed French–Mooré New Year message for alignment."""
    from moore_web.new_year_message import prepare_new_year_pair

    parallel = prepare_new_year_pair(collection_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(msgspec.json.encode(parallel))
    typer.echo(f"FR: {len(parallel.french)} segments  MO: {len(parallel.moore)} segments → {output}")


# TODO: replace Kadé by Poko and Katiu, Atega too


def _load_kade_book(path: Path):
    from moore_web.book_parser_facilitateur import parse_book_from_json

    return parse_book_from_json(str(path))


def _write_aligned(aligned, out: Path, use_jsonl: bool) -> None:
    if use_jsonl:
        written = aligned.write_jsonl(str(out))
        if len(written) > 1:
            typer.echo(f"Wrote {len(aligned.french)} aligned pairs across {len(written)} files:")
            for w in written:
                typer.echo(f"  → {w}")
            return
    else:
        out.write_bytes(msgspec.json.encode(aligned))
    typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")


def _finalize_aligned(
    aligned,
    out,  # str | Path
    jsonl: bool,
    hf_private: bool,
    add_lang_id: bool,
    add_consistency: bool,
    add_quality_warn: bool,
    add_len_ratio: bool,
    add_laser_score: bool,
    add_comet_qe: bool,
    comet_batch_size: int = 8,
    postprocess: Callable[[list[dict]], list[dict]] | None = None,
) -> None:
    """Write aligned corpus, optionally annotating and/or pushing to HF Hub."""
    out_str = str(out)
    needs_annotation = any(
        [add_lang_id, add_consistency, add_quality_warn, add_len_ratio, add_laser_score, add_comet_qe]
    )
    is_hf = out_str.startswith("hf://")

    if needs_annotation or is_hf or postprocess:
        from datasets import Dataset

        from moore_web import annotate as _ann
        from moore_web.flatten import flat_rows_to_long

        # Postprocess (lexicon synonym-splitting/proverb-note cleanup) runs on
        # the flat french/moore shape it expects; convert to the long-format
        # HF schema (one row per language pair) only after that's done.
        rows = [
            {"french": f, "moore": m, "laser_score": s}
            for f, m, s in zip(aligned.french, aligned.moore, aligned.scores)
        ]
        if aligned.english:
            for row, en in zip(rows, aligned.english):
                row["english"] = en
        if aligned.doc_ids:
            for row, doc_id in zip(rows, aligned.doc_ids):
                row["doc_id"] = doc_id

        if postprocess:
            rows = postprocess(rows)

        rows = flat_rows_to_long(rows, aligned.source)

        dataset = Dataset.from_list(rows)

        if needs_annotation:
            # score_dataset (score_laser.py) reads src_lang/tgt_lang per row
            # from the dataset's own src_lang/tgt_lang columns when neither is
            # passed explicitly, so a dataset with mixed pairs (e.g. simple's
            # interspersed mos-fra and mos-eng rows) is scored correctly
            # without needing to pick one language pair here.
            dataset = _ann.annotate(
                dataset,
                src_field="source_text",
                tgt_field="target_text",
                lang_id=add_lang_id,
                quality_warn=add_quality_warn,
                consistency=add_consistency,
                len_ratio=add_len_ratio,
                laser=add_laser_score,
                comet_qe=add_comet_qe,
                comet_batch_size=comet_batch_size,
            )
            if not add_quality_warn and "quality_warnings" in dataset.column_names:
                dataset = dataset.remove_columns(["quality_warnings"])
            if not add_consistency and "identification_consistency" in dataset.column_names:
                dataset = dataset.remove_columns(["identification_consistency"])

        _ann.save_data(dataset, out_str, private=hf_private)
    else:
        _write_aligned(aligned, Path(out_str), jsonl)


def _dedup_aligned(aligned, comet_batch_size: int = 8):
    """Deduplicate an AlignedCorpus using COMET-QE and return a new one."""
    from moore_web.dedup_aligned_comet import deduplicate_by_comet
    from moore_web.flatten import AlignedCorpus

    has_doc_ids = bool(aligned.doc_ids)
    pairs = [
        {"fr": f, "mo": m, "laser_score": s} for f, m, s in zip(aligned.french, aligned.moore, aligned.scores)
    ]
    if has_doc_ids:
        for p, doc_id in zip(pairs, aligned.doc_ids):
            p["doc_id"] = doc_id
    typer.echo("      Running COMET-QE deduplication…")
    pairs = deduplicate_by_comet(pairs, batch_size=comet_batch_size)
    return AlignedCorpus(
        french=[p["fr"] for p in pairs],
        moore=[p["mo"] for p in pairs],
        scores=[p["laser_score"] for p in pairs],
        doc_ids=[p["doc_id"] for p in pairs] if has_doc_ids else [],
        source=aligned.source,
    )


def _align_per_unit(unit_parallels, min_score: float, source: str):
    """Align each unit's sentences independently with LASER + FastDTW and concatenate.

    For sources whose units are already known to correspond, so FastDTW never
    has to guess correspondence across unit boundaries.
    """
    from laser_encoders import LaserEncoderPipeline

    from moore_web.align_corpus import align_from_embeddings as _align_from_embs
    from moore_web.flatten import AlignedCorpus

    laser_fr = LaserEncoderPipeline(lang="fra")
    laser_mo = LaserEncoderPipeline(lang="mos")
    all_fr_embs = laser_fr.encode_sentences(
        [s for _, dp in unit_parallels for s in dp.french], normalize_embeddings=True
    )
    all_mo_embs = laser_mo.encode_sentences(
        [s for _, dp in unit_parallels for s in dp.moore], normalize_embeddings=True
    )

    all_fr, all_mo, all_scores, all_doc_ids = [], [], [], []
    fr_offset = mo_offset = 0
    for unit_id, dp in unit_parallels:
        fr_end, mo_end = fr_offset + len(dp.french), mo_offset + len(dp.moore)
        aligned_dp = _align_from_embs(
            dp, all_fr_embs[fr_offset:fr_end], all_mo_embs[mo_offset:mo_end], min_score=min_score
        )
        fr_offset, mo_offset = fr_end, mo_end
        all_fr.extend(aligned_dp.french)
        all_mo.extend(aligned_dp.moore)
        all_scores.extend(aligned_dp.scores)
        all_doc_ids.extend([unit_id] * len(aligned_dp.french))

    return AlignedCorpus(french=all_fr, moore=all_mo, scores=all_scores, doc_ids=all_doc_ids, source=source)


# Default page ranges for Kadé PDFs (content pages only, excludes front/back matter).
# Not exposed as CLI options for now — override by calling _parse_kade_file directly.
_KADE_PAGE_RANGES: dict[KadeLang, tuple[int, int]] = {
    KadeLang.french: (3, 57),
    KadeLang.moore: (3, 55),
}


def _parse_kade_file(
    input_path: Path,
    lang: KadeLang,
    page_range: Optional[tuple[int, int]] = None,
):
    """Parse a single Kadé PDF or TXT file and return a Book."""
    import re as _re

    from moore_web.book_parser_facilitateur import (
        FRENCH_INTRO_SECTION_TITLES,
        FRENCH_INTRO_SUBSECTION_TITLES,
        MOORE_INTRO_SECTION_TITLES,
        MOORE_INTRO_SUBSECTION_TITLES,
        MOORE_SECTION_TITLES,
        SECTION_TITLES,
        parse_with_chapters,
    )

    # Build section regexes from titles
    if lang == KadeLang.moore:
        sec_titles = MOORE_SECTION_TITLES
        intro_titles = MOORE_INTRO_SECTION_TITLES
        intro_sub_titles = MOORE_INTRO_SUBSECTION_TITLES
        intro_sub_key = "Sẽn n kẽed ne seb kãngã"
        stop_title = "Tʋʋm teedo"
    else:
        sec_titles = SECTION_TITLES
        intro_titles = FRENCH_INTRO_SECTION_TITLES
        intro_sub_titles = FRENCH_INTRO_SUBSECTION_TITLES
        intro_sub_key = "Comment utiliser ce manuel"
        stop_title = "Matériels de formation"

    stop_before_re = _re.compile(_re.escape(stop_title), _re.IGNORECASE)

    sec_patterns = [_re.compile(_re.escape(t), _re.IGNORECASE) for t in sec_titles]
    intro_patterns = [_re.compile(_re.escape(t), _re.IGNORECASE) for t in intro_titles]
    intro_sub_patterns = [
        _re.compile(r"(?:\d+\.\s+)?" + _re.escape(t) + r"\s*$", _re.IGNORECASE) for t in intro_sub_titles
    ]
    intro_sub_map = {intro_sub_key: (intro_sub_patterns, intro_sub_titles)}

    if input_path.suffix.lower() == ".pdf":
        from moore_web.pdf_extractor import extract_pdf_blocks

        effective_range = page_range if page_range is not None else _KADE_PAGE_RANGES[lang]
        text = extract_pdf_blocks(str(input_path), page_range=effective_range)
    else:
        text = input_path.read_text(encoding="utf-8")

    return parse_with_chapters(
        text,
        sec_patterns,
        sec_titles,
        intro_section_regexes=intro_patterns,
        intro_section_titles=intro_titles,
        intro_subsection_map=intro_sub_map,
        stop_before=stop_before_re,
    )


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


@app.command()
def parse(
    source: Annotated[Source, typer.Option("--source", "-s", help="Source type.")] = Source.sida,
    # sida / news
    input: Annotated[
        Optional[Path],
        typer.Option(
            "--input", "-i", exists=True, dir_okay=False, help="Input file (sida PDF or news JSON)."
        ),
    ] = None,
    # kade only
    kade_input: Annotated[
        Optional[Path],
        typer.Option(
            "--kade-input",
            exists=True,
            dir_okay=False,
            help="Kadé PDF or TXT to parse (kade only). Use twice: once per language.",
        ),
    ] = None,
    lang: Annotated[
        Optional[KadeLang],
        typer.Option("--lang", "-l", help="Language of the kade input file (kade only)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output JSON path (default: derived from input)."),
    ] = None,
    # news-specific
    lang_id: Annotated[
        bool,
        typer.Option("--lang-id/--no-lang-id", help="Run language ID annotation (news only)."),
    ] = True,
) -> None:
    """Parse source document(s) to structured JSON.

    [bold]sida:[/bold]    moore-web parse -s sida -i book.pdf -o parsed.json
    [bold]kade:[/bold]    moore-web parse -s kade --kade-input fr.pdf -l french -o fr.json
    [bold]news:[/bold]    moore-web parse -s news -i corpus.json -o segmented.json
    [bold]simple:[/bold]  moore-web parse -s simple -i dict.pdf -o parsed.json
    """
    if source == Source.sida:
        if input is None:
            _err("--input is required for source 'sida'.")
            raise typer.Exit(1)
        out = output or _default_output(input, "_parsed.json")
        from moore_web.book_parser import parse_pdf_to_json

        typer.echo(f"Parsing SIDA book: {input}")
        chapters = parse_pdf_to_json(str(input))
        out.write_bytes(msgspec.json.encode(chapters))
        typer.echo(f"Wrote {len(chapters)} chapters → {out}")

    elif source == Source.kade:
        if kade_input is None or lang is None:
            _err("--kade-input and --lang are required for source 'kade'.")
            raise typer.Exit(1)
        out = output or _default_output(kade_input, f"_{lang.value}_parsed.json")
        typer.echo(f"Parsing Kadé [{lang.value}]: {kade_input}")
        book = _parse_kade_file(kade_input, lang)
        out.write_bytes(msgspec.json.encode(book))
        n = len(book.chapters)
        typer.echo(f"Wrote {n} chapters → {out}")

    elif source == Source.news:
        if input is None:
            _err("--input is required for source 'news'.")
            raise typer.Exit(1)
        out = output or _default_output(input, "_segmented.json")
        typer.echo(f"Parsing news corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))

        if lang_id:
            from moore_web.glotlid import annotate_text_units

            typer.echo("Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {len(corpus)} entries → {out}")

    elif source in (Source.simple, Source.one_column_dict):
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.one_column_dict_parser import parse_doc

        out = output or _default_output(input, "_parsed.json")
        typer.echo(f"Parsing simple dictionary: {input}")
        with pymupdf.open(str(input)) as doc:
            pages = parse_doc(doc)
        out.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        n = sum(len(p) for p in pages)
        typer.echo(f"Wrote {n} entries across {len(pages)} pages → {out}")


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


@app.command()
def flatten(
    source: Annotated[Source, typer.Option("--source", "-s", help="Source type.")] = Source.sida,
    # sida / news / simple
    input: Annotated[
        Optional[Path],
        typer.Option(
            "--input", "-i", exists=True, dir_okay=False, help="Parsed JSON (sida, news, or simple)."
        ),
    ] = None,
    # kade only
    fr_input: Annotated[
        Optional[Path],
        typer.Option("--fr-input", exists=True, dir_okay=False, help="Parsed French Book JSON (kade only)."),
    ] = None,
    mo_input: Annotated[
        Optional[Path],
        typer.Option("--mo-input", exists=True, dir_okay=False, help="Parsed Mooré Book JSON (kade only)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output ParallelText JSON (default: derived from input)."),
    ] = None,
    segment: Annotated[
        bool,
        typer.Option("--segment/--no-segment", help="Sentence-segment each text block."),
    ] = True,
    # simple only
    examples: Annotated[
        bool,
        typer.Option("--examples/--no-examples", help="Include pre-aligned example triplets (simple only)."),
    ] = True,
    entries: Annotated[
        bool,
        typer.Option(
            "--entries/--no-entries", help="Include definition entries as moore/fr/en rows (simple only)."
        ),
    ] = False,
) -> None:
    """Flatten parsed JSON to a ParallelText sentence list.

    [bold]sida:[/bold]    moore-web flatten -s sida -i parsed.json -o parallel.json
    [bold]kade:[/bold]    moore-web flatten -s kade --fr-input fr.json --mo-input mo.json -o parallel.json
    [bold]news:[/bold]    moore-web flatten -s news -i segmented.json -o parallel.json
    [bold]simple:[/bold]  moore-web flatten -s simple -i parsed.json -o parallel.json
    """
    from moore_web.flatten import (
        flatten_facilitateur_pair,
        flatten_news_entries,
        flatten_sida_book,
    )

    if source == Source.sida:
        if input is None:
            _err("--input is required for source 'sida'.")
            raise typer.Exit(1)
        from moore_web.book_parser import Chapter

        chapters = msgspec.json.decode(input.read_bytes(), type=list[Chapter])
        parallel = flatten_sida_book(chapters, segment=segment)
        out = output or _default_output(input, "_parallel.json")

    elif source == Source.kade:
        if fr_input is None or mo_input is None:
            _err("--fr-input and --mo-input are required for source 'kade'.")
            raise typer.Exit(1)
        fr_book = _load_kade_book(fr_input)
        mo_book = _load_kade_book(mo_input)
        parallel = flatten_facilitateur_pair(fr_book, mo_book, segment=segment)
        out = output or fr_input.with_name("kade_parallel.json")

    elif source == Source.news:
        if input is None:
            _err("--input is required for source 'news'.")
            raise typer.Exit(1)
        entries = json.loads(input.read_text(encoding="utf-8"))
        parallel = flatten_news_entries(entries, segment=segment)
        out = output or _default_output(input, "_parallel.json")

    elif source in (Source.simple, Source.one_column_dict):
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        from moore_web.flatten import flatten_simple_parser

        pages = json.loads(input.read_text(encoding="utf-8"))
        parallel = flatten_simple_parser(pages, include_examples=examples, include_entries=entries)
        out = output or _default_output(input, "_parallel.json")

    elif source == Source.conseils:
        if input is None:
            _err("--input is required for source 'conseils'.")
            raise typer.Exit(1)
        from moore_web.flatten import ParallelText, flatten_conseils

        corpus = json.loads(input.read_text(encoding="utf-8"))
        date_parallels = flatten_conseils(corpus, segment=segment)
        # Merge all dates into one ParallelText for the flatten command
        parallel = ParallelText(source="conseils")
        for _date, dp in date_parallels:
            parallel.french.extend(dp.french)
            parallel.moore.extend(dp.moore)
        out = output or _default_output(input, "_parallel.json")
        typer.echo(f"Flattened {len(date_parallels)} sessions.")

    out.write_bytes(msgspec.json.encode(parallel))
    typer.echo(
        f"FR: {len(parallel.french)} sentences  MO: {len(parallel.moore)} sentences"
        + (f"  EN: {len(parallel.english)}" if parallel.english else "")
        + f" → {out}"
    )


# ---------------------------------------------------------------------------
# parse-flat
# ---------------------------------------------------------------------------


@app.command(name="parse-flat")
def parse_flat(
    source: Annotated[Source, typer.Option("--source", "-s", help="Source type.")] = Source.sida,
    input: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            exists=True,
            dir_okay=False,
            help="Input file (sida PDF, news JSON, or simple PDF).",
        ),
    ] = None,
    fr_input: Annotated[
        Optional[Path],
        typer.Option("--fr-input", exists=True, dir_okay=False, help="French PDF/TXT (kade only)."),
    ] = None,
    mo_input: Annotated[
        Optional[Path],
        typer.Option("--mo-input", exists=True, dir_okay=False, help="Mooré PDF/TXT (kade only)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output ParallelText JSON."),
    ] = None,
    segment: Annotated[
        bool,
        typer.Option("--segment/--no-segment", help="Sentence-segment each text block."),
    ] = True,
    lang_id: Annotated[
        bool,
        typer.Option("--lang-id/--no-lang-id", help="Run language ID annotation (news only)."),
    ] = True,
    examples: Annotated[
        bool,
        typer.Option("--examples/--no-examples", help="Include pre-aligned example triplets (simple only)."),
    ] = True,
    entries: Annotated[
        bool,
        typer.Option(
            "--entries/--no-entries", help="Include definition entries as moore/fr/en rows (simple only)."
        ),
    ] = False,
) -> None:
    """Parse and flatten in one step.

    [bold]sida:[/bold]    moore-web parse-flat -s sida -i book.pdf -o parallel.json
    [bold]kade:[/bold]    moore-web parse-flat -s kade --fr-input fr.pdf --mo-input mo.pdf -o parallel.json
    [bold]news:[/bold]    moore-web parse-flat -s news -i corpus.json -o parallel.json
    [bold]simple:[/bold]  moore-web parse-flat -s simple -i dict.pdf -o parallel.json
    """
    from moore_web.flatten import (
        flatten_facilitateur_pair,
        flatten_news_entries,
        flatten_sida_book,
    )

    if source == Source.sida:
        if input is None:
            _err("--input is required for source 'sida'.")
            raise typer.Exit(1)
        from moore_web.book_parser import parse_pdf_to_json

        typer.echo(f"Parsing SIDA book: {input}")
        chapters = parse_pdf_to_json(str(input))
        parallel = flatten_sida_book(chapters, segment=segment)
        out = output or _default_output(input, "_parallel.json")

    elif source == Source.kade:
        if fr_input is None or mo_input is None:
            _err("--fr-input and --mo-input are required for source 'kade'.")
            raise typer.Exit(1)
        typer.echo(f"Parsing Kadé FR: {fr_input}")
        fr_book = _parse_kade_file(fr_input, KadeLang.french)
        typer.echo(f"Parsing Kadé MO: {mo_input}")
        mo_book = _parse_kade_file(mo_input, KadeLang.moore)
        parallel = flatten_facilitateur_pair(fr_book, mo_book, segment=segment)
        out = output or fr_input.with_name("kade_parallel.json")

    elif source == Source.news:
        if input is None:
            _err("--input is required for source 'news'.")
            raise typer.Exit(1)
        typer.echo(f"Parsing news corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))

        if lang_id:
            from moore_web.glotlid import annotate_text_units

            typer.echo("Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        parallel = flatten_news_entries(corpus, segment=segment)
        out = output or _default_output(input, "_parallel.json")

    elif source in (Source.simple, Source.one_column_dict):
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.flatten import flatten_simple_parser
        from moore_web.one_column_dict_parser import parse_doc

        typer.echo(f"Parsing simple dictionary: {input}")
        with pymupdf.open(str(input)) as doc:
            pages = parse_doc(doc)
        parallel = flatten_simple_parser(pages, include_examples=examples, include_entries=entries)
        out = output or _default_output(input, "_parallel.json")

    out.write_bytes(msgspec.json.encode(parallel))
    typer.echo(
        f"FR: {len(parallel.french)} sentences  MO: {len(parallel.moore)} sentences"
        + (f"  EN: {len(parallel.english)}" if parallel.english else "")
        + f" → {out}"
    )


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


@app.command()
def align(
    input: Annotated[Path, typer.Argument(exists=True, help="ParallelText JSON to align.")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output aligned JSON (default: derived from input)."),
    ] = None,
    min_score: Annotated[
        float,
        typer.Option(
            "--min-laser-score", min=0.0, max=1.0, help="Drop pairs below this LASER cosine similarity."
        ),
    ] = 0.0,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl/--json", help="Write output as JSONL (default) or a single JSON file."),
    ] = True,
) -> None:
    """Align a ParallelText JSON using LASER embeddings + FastDTW.

    Example: moore-web align parallel.json -o aligned.jsonl --min-laser-score 0.6
    """
    from moore_web.align_corpus import align as _align
    from moore_web.flatten import ParallelText

    suffix = "_aligned.jsonl" if jsonl else "_aligned.json"
    out = output or _default_output(input, suffix)

    parallel = ParallelText.from_json(input.read_bytes())
    typer.echo(f"Input: {len(parallel.french)} FR  {len(parallel.moore)} MO")

    aligned = _align(parallel, min_score=min_score)

    if jsonl:
        aligned.write_jsonl(str(out))
    else:
        out.write_bytes(msgspec.json.encode(aligned))
    typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


@app.command()
def annotate(
    input: Annotated[str, typer.Option("--input", "-i", help="Local JSONL or hf://owner/repo.")],
    output: Annotated[str, typer.Option("--output", "-o", help="Local JSONL or hf://owner/repo.")],
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            help="Config to load (hf:// input only) -- required when the repo has more than "
            "one, e.g. a language pair from a repo save_data split by pair (\"mos-fra\").",
        ),
    ] = None,
    src: Annotated[str, typer.Option("--src", help="Source field name in the dataset.")] = "french",
    tgt: Annotated[str, typer.Option("--tgt", help="Target field name in the dataset.")] = "moore",
    lang_id: Annotated[
        bool, typer.Option("--lang-id", is_flag=True, help="Add GlotLID language-ID scores.")
    ] = False,
    consistency: Annotated[
        bool, typer.Option("--consistency", is_flag=True, help="Add identification_consistency score.")
    ] = False,
    quality_warn: Annotated[
        bool, typer.Option("--quality-warn", is_flag=True, help="Add quality_warnings list.")
    ] = False,
    len_ratio: Annotated[
        bool, typer.Option("--len-ratio", is_flag=True, help="Add len_ratio (character-length ratio).")
    ] = False,
    laser_score: Annotated[
        bool, typer.Option("--laser-score", is_flag=True, help="Add LASER cosine similarity.")
    ] = False,
    src_lang: Annotated[
        Optional[str],
        typer.Option(
            "--src-lang",
            help="LASER language code for the source encoder (e.g. fra, eng). "
            "Inferred from --src when known; required for unrecognized fields.",
        ),
    ] = None,
    tgt_lang: Annotated[
        Optional[str],
        typer.Option(
            "--tgt-lang",
            help="LASER language code for the target encoder (e.g. mos). "
            "Inferred from --tgt when known; required for unrecognized fields.",
        ),
    ] = None,
    comet_qe: Annotated[
        bool, typer.Option("--comet-qe", is_flag=True, help="Add COMET-QE translation quality score.")
    ] = False,
    all_annotations: Annotated[
        bool, typer.Option("--all", is_flag=True, help="Enable all annotation flags.")
    ] = False,
    hf_private: Annotated[
        bool, typer.Option("--hf-private", is_flag=True, help="Push to HuggingFace as private dataset.")
    ] = False,
) -> None:
    """Enrich an aligned dataset with quality signals.

    All annotation flags are off by default — opt in to what you need.

    [bold]Local:[/bold]  moore-web annotate -i data.jsonl -o out.jsonl --consistency --quality-warn
    [bold]All:[/bold]    moore-web annotate -i data.jsonl -o out.jsonl --all
    [bold]HF:[/bold]     moore-web annotate -i hf://owner/src -o hf://owner/dst --all
    [bold]HF config:[/bold] moore-web annotate -i hf://owner/src --config mos-fra -o hf://owner/dst --all
    """
    from moore_web import annotate as _ann

    if all_annotations:
        lang_id = consistency = quality_warn = len_ratio = laser_score = comet_qe = True

    if not any([lang_id, consistency, quality_warn, len_ratio, laser_score, comet_qe]):
        _err(
            "No annotation flags specified. Pass at least one of: --lang-id, --consistency, "
            "--quality-warn, --len-ratio, --laser-score, --comet-qe."
        )
        raise typer.Exit(1)

    dataset = _ann.load_data(input, config_name=config)
    dataset = _ann.annotate(
        dataset,
        src_field=src,
        tgt_field=tgt,
        lang_id=lang_id,
        quality_warn=quality_warn,
        consistency=consistency,
        len_ratio=len_ratio,
        laser=laser_score,
        comet_qe=comet_qe,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    # Drop the column not requested when only one of the shared pair is selected.
    if not quality_warn and "quality_warnings" in dataset.column_names:
        dataset = dataset.remove_columns(["quality_warnings"])
    if not consistency and "identification_consistency" in dataset.column_names:
        dataset = dataset.remove_columns(["identification_consistency"])

    _ann.save_data(dataset, output, private=hf_private)


# ---------------------------------------------------------------------------
# clean-lexicon
# ---------------------------------------------------------------------------


@app.command(name="clean-lexicon")
def clean_lexicon(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            dir_okay=False,
            help="Lexicon JSONL file to clean (modified in-place).",
        ),
    ],
    split_synonyms: Annotated[
        bool,
        typer.Option(
            "--split-synonyms",
            is_flag=True,
            help=(
                "Explode comma/semicolon-separated synonym entries into one row per pair. "
                "Warns when Moore contains a comma (parsing issue) or FR/EN counts mismatch."
            ),
        ),
    ] = False,
    strip_proverb_notes: Annotated[
        bool,
        typer.Option(
            "--strip-proverb-notes",
            is_flag=True,
            help=(
                "Remove parenthetical proverb explanations — e.g. '(Proverbe: …)' — "
                "and leading labels such as 'Proverbe :' from french and english fields. "
                "len_ratio is recalculated when present."
            ),
        ),
    ] = False,
) -> None:
    """Clean a lexicon JSONL file in-place.

    [bold]Split synonyms:[/bold]
        moore-web clean-lexicon -i lexicon.jsonl --split-synonyms

    [bold]Strip proverb notes:[/bold]
        moore-web clean-lexicon -i lexicon.jsonl --strip-proverb-notes

    [bold]Both:[/bold]
        moore-web clean-lexicon -i lexicon.jsonl --split-synonyms --strip-proverb-notes
    """
    if not split_synonyms and not strip_proverb_notes:
        _err("No flags specified. Pass --split-synonyms and/or --strip-proverb-notes.")
        raise typer.Exit(1)

    from moore_web.clean_lexicon import process as _clean

    with open(input, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]

    output, n_split, n_proverb = _clean(
        entries=entries,
        split_synonyms=split_synonyms,
        strip_proverb_notes=strip_proverb_notes,
    )

    with open(input, "w", encoding="utf-8") as fh:
        for e in output:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    typer.echo(f"Input:  {len(entries)} entries")
    if split_synonyms:
        typer.echo(f"Split:  +{n_split} entries from synonym lists")
    if strip_proverb_notes:
        typer.echo(f"Stripped proverb notes: {n_proverb} entries")
    typer.echo(f"Output: {len(output)} entries → {input}")


# ---------------------------------------------------------------------------
# e2e
# ---------------------------------------------------------------------------


@app.command()
def e2e(
    source: Annotated[Source, typer.Option("--source", "-s", help="Source type.")] = Source.sida,
    input: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            exists=True,
            help=(
                "Input file (sida PDF, news/conseils JSON, or simple PDF), or the archived "
                "app directory for moore-tales. "
                "Must be a local path. For news/conseils corpora hosted on HuggingFace, "
                "download the file first: "
                "huggingface-cli download owner/repo corpus.json --repo-type dataset --local-dir ."
            ),
        ),
    ] = None,
    fr_input: Annotated[
        Optional[Path],
        typer.Option(
            "--fr-input",
            exists=True,
            dir_okay=False,
            help="French PDF/TXT (kade, digital, udhr, abc-coepouses).",
        ),
    ] = None,
    mo_input: Annotated[
        Optional[Path],
        typer.Option(
            "--mo-input",
            exists=True,
            dir_okay=False,
            help="Mooré PDF/TXT (kade, digital, udhr, abc-coepouses).",
        ),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output path or hf://owner/repo (default: derived from input)."),
    ] = None,
    segment: Annotated[
        bool,
        typer.Option("--segment/--no-segment", help="Sentence-segment each text block."),
    ] = True,
    min_score: Annotated[
        float,
        typer.Option(
            "--min-laser-score", min=0.0, max=1.0, help="Drop pairs below this LASER cosine similarity."
        ),
    ] = 0.0,
    lang_id: Annotated[
        bool,
        typer.Option("--lang-id/--no-lang-id", help="Run language ID annotation (news only)."),
    ] = True,
    examples: Annotated[
        bool,
        typer.Option("--examples/--no-examples", help="Include pre-aligned example triplets (simple only)."),
    ] = True,
    entries: Annotated[
        bool,
        typer.Option(
            "--entries/--no-entries", help="Include definition entries as moore/fr/en rows (simple only)."
        ),
    ] = False,
    entries_output: Annotated[
        Optional[Path],
        typer.Option(
            "--entries-output",
            help="Write entries to a separate file (simple only). Avoids parsing the PDF twice.",
        ),
    ] = None,
    terms: Annotated[
        bool,
        typer.Option("--terms/--no-terms", help="Include term pairs fr_term/mos_term (digital only)."),
    ] = True,
    definitions: Annotated[
        bool,
        typer.Option(
            "--definitions/--no-definitions",
            help="Include definition pairs fr_definition/mos_definition (digital only).",
        ),
    ] = False,
    definitions_output: Annotated[
        Optional[Path],
        typer.Option(
            "--definitions-output",
            help="Write definition pairs to a separate file (digital only). Avoids parsing the PDFs twice.",
        ),
    ] = None,
    drop_duplicate: Annotated[
        bool,
        typer.Option(
            "--drop-duplicate/--no-drop-duplicate",
            help="Deduplicate aligned pairs with COMET-QE, keeping highest score per group (not available for simple).",
        ),
    ] = False,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl/--json", help="Write output as JSONL (default) or a single JSON file."),
    ] = True,
    add_lang_id: Annotated[
        bool,
        typer.Option("--add-lang-id", is_flag=True, help="Annotate aligned output with GlotLID scores."),
    ] = False,
    add_consistency: Annotated[
        bool,
        typer.Option(
            "--add-consistency", is_flag=True, help="Annotate aligned output with identification_consistency."
        ),
    ] = False,
    add_quality_warn: Annotated[
        bool,
        typer.Option(
            "--add-quality-warn", is_flag=True, help="Annotate aligned output with quality_warnings."
        ),
    ] = False,
    add_len_ratio: Annotated[
        bool,
        typer.Option("--add-len-ratio", is_flag=True, help="Annotate aligned output with len_ratio."),
    ] = False,
    add_laser_score: Annotated[
        bool,
        typer.Option(
            "--add-laser-score", is_flag=True, help="Annotate aligned output with LASER similarity."
        ),
    ] = False,
    add_comet_qe: Annotated[
        bool,
        typer.Option("--add-comet-qe", is_flag=True, help="Annotate aligned output with COMET-QE score."),
    ] = False,
    comet_batch_size: Annotated[
        int,
        typer.Option(
            "--comet-batch-size",
            min=1,
            help="COMET-QE batch size for --drop-duplicate and --add-comet-qe; lower it on out-of-memory.",
        ),
    ] = 8,
    do_annotate: Annotated[
        bool,
        typer.Option("--annotate", is_flag=True, help="Shorthand: enable all --add-* annotation flags."),
    ] = False,
    hf_private: Annotated[
        bool,
        typer.Option("--hf-private", is_flag=True, help="Push to HuggingFace as private dataset."),
    ] = False,
    split_synonyms: Annotated[
        bool,
        typer.Option(
            "--split-synonyms",
            is_flag=True,
            help="(simple only) Explode comma/semicolon-separated synonym entries into one row per pair.",
        ),
    ] = False,
    strip_proverb_notes: Annotated[
        bool,
        typer.Option(
            "--strip-proverb-notes",
            is_flag=True,
            help="(simple only) Strip parenthetical proverb explanations and leading 'Proverbe :' labels.",
        ),
    ] = False,
) -> None:
    """End-to-end pipeline: parse → flatten → align.

    [bold]sida:[/bold]              moore-web e2e -s sida -i book.pdf -o aligned.jsonl
    [bold]kade:[/bold]              moore-web e2e -s kade --fr-input fr.pdf --mo-input mo.pdf -o aligned.jsonl
    [bold]news:[/bold]              moore-web e2e -s news -i corpus.json -o aligned.jsonl
    [bold]simple:[/bold]            moore-web e2e -s simple -i dict.pdf -o aligned.jsonl
    [bold]digital (terms):[/bold]   moore-web e2e -s digital --fr-input lexique.pdf --mo-input glossaire.pdf -o terms.jsonl
    [bold]digital (both):[/bold]    moore-web e2e -s digital --fr-input lexique.pdf --mo-input glossaire.pdf -o terms.jsonl --definitions-output defs.jsonl --add-laser-score --add-comet-qe --add-quality-warn
    [bold]udhr:[/bold]              moore-web e2e -s udhr --fr-input udhr-fra.txt --mo-input udhr-mos.txt -o udhr.jsonl
    [bold]abc-coepouses:[/bold]     moore-web e2e -s abc-coepouses --fr-input 266-les-couses.txt --mo-input 267-les-co-epouses-moore.txt -o tale.jsonl
    [bold]moore-tales:[/bold]       moore-web e2e -s moore-tales -i apps/mos-contes-volume-5 -o tales.jsonl
    [bold]HF output:[/bold]         moore-web e2e -s sida -i book.pdf -o hf://owner/repo --annotate
    """
    if do_annotate:
        add_lang_id = add_consistency = add_quality_warn = add_len_ratio = add_laser_score = add_comet_qe = (
            True
        )

    if (split_synonyms or strip_proverb_notes) and source not in (Source.simple, Source.one_column_dict):
        _err("--split-synonyms / --strip-proverb-notes are only supported for --source simple or one-column-dict.")
        raise typer.Exit(1)

    if (split_synonyms or strip_proverb_notes) and output and str(output).startswith("hf://"):
        _err("--split-synonyms / --strip-proverb-notes are not supported with HuggingFace output.")
        raise typer.Exit(1)

    if (terms is False or definitions or definitions_output is not None) and source != Source.digital:
        _err("--terms / --definitions / --definitions-output are only supported for --source digital.")
        raise typer.Exit(1)

    _postprocess_entries: Callable[[list[dict]], list[dict]] | None = None
    _postprocess_examples: Callable[[list[dict]], list[dict]] | None = None
    if split_synonyms or strip_proverb_notes:
        from moore_web.clean_lexicon import process as _clean

        if split_synonyms:

            def _postprocess_entries(rows: list[dict]) -> list[dict]:  # type: ignore[misc]
                cleaned, n_split, _ = _clean(rows, split_synonyms=True, strip_proverb_notes=False)
                typer.echo(f"      clean: +{n_split} entries from synonym splitting")
                return cleaned

        if strip_proverb_notes:

            def _postprocess_examples(rows: list[dict]) -> list[dict]:  # type: ignore[misc]
                cleaned, _, n_proverb = _clean(rows, split_synonyms=False, strip_proverb_notes=True)
                typer.echo(f"      clean: {n_proverb} proverb notes stripped")
                return cleaned

    _ann_kwargs: dict = dict(
        add_lang_id=add_lang_id,
        add_consistency=add_consistency,
        add_quality_warn=add_quality_warn,
        add_len_ratio=add_len_ratio,
        add_laser_score=add_laser_score,
        add_comet_qe=add_comet_qe,
        comet_batch_size=comet_batch_size,
        hf_private=hf_private,
    )

    from moore_web.align_corpus import align as _align
    from moore_web.flatten import (
        flatten_facilitateur_pair,
    )

    # ── parse + flatten ──────────────────────────────────────────────────────
    _ext = ".jsonl" if jsonl else ".json"
    if source == Source.sida:
        if input is None:
            _err("--input is required for source 'sida'.")
            raise typer.Exit(1)
        from moore_web.book_parser import parse_pdf_to_json
        from moore_web.flatten import AlignedCorpus, flatten_sida_book_per_unit

        typer.echo(f"[1/3] Parsing SIDA book: {input}")
        chapters = parse_pdf_to_json(str(input))
        typer.echo(f"[2/3] Flattening {len(chapters)} chapters…")
        unit_parallels = flatten_sida_book_per_unit(chapters, segment=segment)
        out = output or _default_output(input, f"_aligned{_ext}")

        # The PDF is laid out in strict left/right (Mooré/French) columns on
        # every page, so a page's two languages are already known to
        # correspond. Align each page/enum-item independently instead of
        # over the whole flattened book — this keeps FastDTW's monotonic
        # path from having to guess correspondence across 45 pages at once.
        typer.echo("[3/3] Aligning per page with LASER + FastDTW…")
        from laser_encoders import LaserEncoderPipeline

        from moore_web.align_corpus import align_from_embeddings as _align_from_embs

        laser_fr = LaserEncoderPipeline(lang="fra")
        laser_mo = LaserEncoderPipeline(lang="mos")

        all_fr_sents = [s for _, dp in unit_parallels for s in dp.french]
        all_mo_sents = [s for _, dp in unit_parallels for s in dp.moore]
        all_fr_embs = laser_fr.encode_sentences(all_fr_sents, normalize_embeddings=True)
        all_mo_embs = laser_mo.encode_sentences(all_mo_sents, normalize_embeddings=True)

        all_fr, all_mo, all_scores, all_doc_ids = [], [], [], []
        fr_offset = mo_offset = 0
        for unit_id, dp in unit_parallels:
            fr_end, mo_end = fr_offset + len(dp.french), mo_offset + len(dp.moore)
            aligned_dp = _align_from_embs(
                dp, all_fr_embs[fr_offset:fr_end], all_mo_embs[mo_offset:mo_end], min_score=min_score
            )
            fr_offset, mo_offset = fr_end, mo_end
            all_fr.extend(aligned_dp.french)
            all_mo.extend(aligned_dp.moore)
            all_scores.extend(aligned_dp.scores)
            all_doc_ids.extend([unit_id] * len(aligned_dp.french))

        aligned = AlignedCorpus(
            french=all_fr,
            moore=all_mo,
            scores=all_scores,
            doc_ids=all_doc_ids,
            source="sida-bilingual-book",
        )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source == Source.kade:
        if fr_input is None or mo_input is None:
            _err("--fr-input and --mo-input are required for source 'kade'.")
            raise typer.Exit(1)
        typer.echo(f"[1/3] Parsing Kadé FR: {fr_input}")
        fr_book = _parse_kade_file(fr_input, KadeLang.french)
        typer.echo(f"[1/3] Parsing Kadé MO: {mo_input}")
        mo_book = _parse_kade_file(mo_input, KadeLang.moore)
        typer.echo("[2/3] Flattening…")
        parallel = flatten_facilitateur_pair(fr_book, mo_book, segment=segment)
        out = output or fr_input.with_name(f"kade_aligned{_ext}")

    elif source == Source.news:
        if input is None:
            _err("--input is required for source 'news'.")
            raise typer.Exit(1)
        typer.echo(f"[1/3] Parsing news corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))

        if lang_id:
            from moore_web.glotlid import annotate_text_units

            typer.echo("      Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.flatten import AlignedCorpus, flatten_news_per_entry
        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        typer.echo("[2/3] Flattening…")
        article_parallels = flatten_news_per_entry(corpus, segment=segment)
        out = output or _default_output(input, f"_aligned{_ext}")
        typer.echo(f"      {len(article_parallels)} bilingual articles found.")

        typer.echo("[3/3] Aligning per article with LASER + FastDTW…")
        from laser_encoders import LaserEncoderPipeline

        from moore_web.align_corpus import align_from_embeddings as _align_from_embs

        laser_fr = LaserEncoderPipeline(lang="fra")
        laser_mo = LaserEncoderPipeline(lang="mos")

        all_fr_sents = [s for _, dp in article_parallels for s in dp.french]
        all_mo_sents = [s for _, dp in article_parallels for s in dp.moore]
        all_fr_embs = laser_fr.encode_sentences(all_fr_sents, normalize_embeddings=True)
        all_mo_embs = laser_mo.encode_sentences(all_mo_sents, normalize_embeddings=True)

        all_fr, all_mo, all_scores, all_doc_ids = [], [], [], []
        fr_offset = mo_offset = 0
        for url, dp in article_parallels:
            fr_end, mo_end = fr_offset + len(dp.french), mo_offset + len(dp.moore)
            aligned_dp = _align_from_embs(
                dp, all_fr_embs[fr_offset:fr_end], all_mo_embs[mo_offset:mo_end], min_score=min_score
            )
            fr_offset, mo_offset = fr_end, mo_end
            all_fr.extend(aligned_dp.french)
            all_mo.extend(aligned_dp.moore)
            all_scores.extend(aligned_dp.scores)
            all_doc_ids.extend([url] * len(aligned_dp.french))

        aligned = AlignedCorpus(
            french=all_fr,
            moore=all_mo,
            scores=all_scores,
            doc_ids=all_doc_ids,
            source="raamde-news",
        )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source in (Source.simple, Source.one_column_dict):
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.flatten import AlignedCorpus, flatten_simple_parser
        from moore_web.one_column_dict_parser import parse_doc

        typer.echo(f"[1/2] Parsing simple dictionary: {input}")
        with pymupdf.open(str(input)) as doc:
            pages = parse_doc(doc)
        typer.echo("[2/2] Flattening…")

        def _write_simple(inc_examples: bool, inc_entries: bool, dest, postprocess=None) -> None:
            p = flatten_simple_parser(pages, include_examples=inc_examples, include_entries=inc_entries)
            typer.echo(f"      FR: {len(p.french)}  MO: {len(p.moore)}  EN: {len(p.english)}  → {dest}")
            a = AlignedCorpus(
                french=p.french,
                moore=p.moore,
                english=p.english,
                scores=[1.0] * len(p.french),
                source=p.source,
            )
            _finalize_aligned(a, dest, jsonl, postprocess=postprocess, **_ann_kwargs)

        out = output or _default_output(input, f"_aligned{_ext}")
        if entries_output is not None:
            # Parse once, write examples and entries to separate files.
            _write_simple(inc_examples=True, inc_entries=False, dest=out, postprocess=_postprocess_examples)
            _write_simple(
                inc_examples=False, inc_entries=True, dest=entries_output, postprocess=_postprocess_entries
            )
        else:
            # Single output — pick the right hook based on what is being written
            if entries and not examples:
                _postprocess = _postprocess_entries
            elif examples and not entries:
                _postprocess = _postprocess_examples
            else:
                _postprocess = None  # mixed output, skip cleaning
            _write_simple(inc_examples=examples, inc_entries=entries, dest=out, postprocess=_postprocess)
        return

    elif source == Source.conseils:
        if input is None:
            _err("--input is required for source 'conseils'.")
            raise typer.Exit(1)
        from moore_web.flatten import AlignedCorpus, flatten_conseils

        typer.echo(f"[1/2] Flattening conseil-des-ministres corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))
        date_parallels = flatten_conseils(corpus, segment=segment)
        out = output or _default_output(input, f"_aligned{_ext}")
        typer.echo(f"      {len(date_parallels)} bilingual sessions found.")

        # Align each date independently, then concatenate.
        typer.echo("[2/2] Aligning per date with LASER + FastDTW…")
        from laser_encoders import LaserEncoderPipeline

        from moore_web.align_corpus import align_from_embeddings as _align_from_embs

        laser_fr = LaserEncoderPipeline(lang="fra")
        laser_mo = LaserEncoderPipeline(lang="mos")

        all_fr_sents = [s for _, dp in date_parallels for s in dp.french]
        all_mo_sents = [s for _, dp in date_parallels for s in dp.moore]
        all_fr_embs = laser_fr.encode_sentences(all_fr_sents, normalize_embeddings=True)
        all_mo_embs = laser_mo.encode_sentences(all_mo_sents, normalize_embeddings=True)

        all_fr, all_mo, all_scores, all_doc_ids = [], [], [], []
        fr_offset = mo_offset = 0
        for date, dp in date_parallels:
            typer.echo(f"      {date}: FR={len(dp.french)}  MO={len(dp.moore)}")
            fr_end, mo_end = fr_offset + len(dp.french), mo_offset + len(dp.moore)
            aligned_dp = _align_from_embs(
                dp, all_fr_embs[fr_offset:fr_end], all_mo_embs[mo_offset:mo_end], min_score=min_score
            )
            fr_offset, mo_offset = fr_end, mo_end
            all_fr.extend(aligned_dp.french)
            all_mo.extend(aligned_dp.moore)
            all_scores.extend(aligned_dp.scores)
            all_doc_ids.extend([date] * len(aligned_dp.french))

        aligned = AlignedCorpus(
            french=all_fr,
            moore=all_mo,
            scores=all_scores,
            doc_ids=all_doc_ids,
            source="conseils",
        )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source == Source.udhr:
        if fr_input is None or mo_input is None:
            _err("--fr-input and --mo-input (UDHR text files) are required for --source udhr.")
            raise typer.Exit(1)

        from moore_web.udhr import pair_udhr_files

        # Paired by article and paragraph position, so there is no LASER step.
        typer.echo("[1/1] Pairing UDHR articles…")
        aligned, skipped = pair_udhr_files(fr_input, mo_input, segment=segment)
        for reason in skipped:
            typer.echo(f"      skipped {reason}")
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        out = output or fr_input.with_name(f"udhr_aligned{_ext}")
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source == Source.abc_coepouses:
        if fr_input is None or mo_input is None:
            _err(
                "--fr-input and --mo-input (archived tale text files) are required for --source abc-coepouses."
            )
            raise typer.Exit(1)

        from moore_web.abc_coepouses_parser import SOURCE as _TALE_SOURCE
        from moore_web.abc_coepouses_parser import parse_abc_coepouses
        from moore_web.flatten import AlignedCorpus, ParallelText

        typer.echo("[1/2] Pairing story beats at the hand-picked anchors…")
        units = parse_abc_coepouses(fr_input, mo_input)
        out = output or fr_input.with_name(f"abc_coepouses_aligned{_ext}")
        if segment:
            typer.echo(
                f"[2/2] Aligning sentences within each of {len(units)} story beats with LASER + FastDTW…"
            )
            unit_parallels = [
                (u.id, ParallelText(french=u.source_sentences, moore=u.target_sentences, source=_TALE_SOURCE))
                for u in units
            ]
            aligned = _align_per_unit(unit_parallels, min_score=min_score, source=_TALE_SOURCE)
        else:
            # The anchors already pair whole story beats; no alignment needed.
            typer.echo(f"[2/2] Keeping {len(units)} story beats as whole pairs…")
            aligned = AlignedCorpus(
                french=[u.source_text for u in units],
                moore=[u.target_text for u in units],
                scores=[None] * len(units),
                doc_ids=[u.id for u in units],
                source=_TALE_SOURCE,
            )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source == Source.moore_tales:
        if input is None or not input.is_dir():
            _err(
                "--input (the archived mos-contes-volume-5 app directory) is required for --source moore-tales."
            )
            raise typer.Exit(1)

        from moore_web.flatten import AlignedCorpus, ParallelText
        from moore_web.moore_tales_parser import COLLECTION_ID as _TALES_SOURCE
        from moore_web.moore_tales_parser import parse_moore_tales

        typer.echo("[1/2] Pairing Mooré and French tale pages by tale number…")
        tales = parse_moore_tales(input)
        out = output or input / f"moore_tales_aligned{_ext}"
        if segment:
            typer.echo(f"[2/2] Aligning sentences within each of {len(tales)} tales with LASER + FastDTW…")
            unit_parallels = [
                (
                    t.id,
                    ParallelText(french=t.target_sentences, moore=t.source_sentences, source=_TALES_SOURCE),
                )
                for t in tales
            ]
            aligned = _align_per_unit(unit_parallels, min_score=min_score, source=_TALES_SOURCE)
        else:
            typer.echo(f"[2/2] Keeping {len(tales)} tales as whole pairs…")
            aligned = AlignedCorpus(
                french=[t.target_text for t in tales],
                moore=[t.source_text for t in tales],
                scores=[None] * len(tales),
                doc_ids=[t.id for t in tales],
                source=_TALES_SOURCE,
            )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned, comet_batch_size)
        _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)
        return

    elif source == Source.digital:
        if fr_input is None or mo_input is None:
            _err(
                "--fr-input (French lexique PDF) and --mo-input (Mooré glossaire PDF) are required for --source digital."
            )
            raise typer.Exit(1)

        from moore_web.flatten import AlignedCorpus
        from moore_web.glossary_parser import (
            align_glossaries,
            extract_french_tables,
            extract_moore_tables,
        )

        typer.echo("[1/2] Extracting glossary tables…")
        moore_entries = extract_moore_tables(str(mo_input))
        typer.echo(f"      Mooré entries : {len(moore_entries)}")
        french_entries = extract_french_tables(str(fr_input))
        typer.echo(f"      French entries: {len(french_entries)}")

        typer.echo("[2/2] Aligning…")
        pairs = align_glossaries(moore_entries, french_entries)
        if moore_entries:
            typer.echo(
                f"      Matched: {len(pairs)} / {len(moore_entries)} ({len(pairs) / len(moore_entries) * 100:.1f}%)"
            )

        def _write_digital(inc_terms: bool, inc_definitions: bool, dest) -> None:
            _Scores = list[float | None]

            if inc_terms and not inc_definitions:
                valid = [p for p in pairs if p.fr_term and p.mos_term]
                fr_texts = [p.fr_term for p in valid]
                mo_texts = [p.mos_term for p in valid]
                # Terms are aligned by exact key match → score 1.0 is appropriate.
                scores = cast(_Scores, [1.0] * len(fr_texts))
                label = "digital-postal-glossary-term"
            elif inc_definitions and not inc_terms:
                valid = [p for p in pairs if p.fr_definition and p.mos_definition]
                fr_texts = [p.fr_definition for p in valid]
                mo_texts = [p.mos_definition for p in valid]
                # Definitions are structurally paired (same glossary entry) but not
                # alignment-scored; use None to signal the score is absent.
                scores = cast(_Scores, [None] * len(fr_texts))
                label = "digital-postal-glossary-term-definition"
            else:
                term_pairs = [p for p in pairs if p.fr_term and p.mos_term]
                def_pairs = [p for p in pairs if p.fr_definition and p.mos_definition]
                fr_texts = [p.fr_term for p in term_pairs] + [p.fr_definition for p in def_pairs]
                mo_texts = [p.mos_term for p in term_pairs] + [p.mos_definition for p in def_pairs]
                scores = cast(_Scores, [1.0] * len(term_pairs) + [None] * len(def_pairs))
                label = "digital-postal-glossary"
            a = AlignedCorpus(
                french=fr_texts,
                moore=mo_texts,
                scores=scores,
                source=label,
            )
            # Terms are exact key matches — LASER/COMET-QE add no information.
            ann = (
                _ann_kwargs
                if not (inc_terms and not inc_definitions)
                else {
                    **_ann_kwargs,
                    "add_laser_score": False,
                    "add_comet_qe": False,
                }
            )
            typer.echo(f"      FR: {len(fr_texts)}  MO: {len(mo_texts)}  → {dest}")
            _finalize_aligned(a, dest, jsonl, **ann)

        out = output or fr_input.with_name(f"digital_aligned{_ext}")
        if definitions_output is not None:
            _write_digital(inc_terms=True, inc_definitions=False, dest=out)
            _write_digital(inc_terms=False, inc_definitions=True, dest=definitions_output)
        else:
            _write_digital(inc_terms=terms, inc_definitions=definitions, dest=out)
        return

    typer.echo(f"      FR: {len(parallel.french)} sentences  MO: {len(parallel.moore)} sentences")

    # ── align ────────────────────────────────────────────────────────────────
    typer.echo("[3/3] Aligning with LASER + FastDTW…")
    aligned = _align(parallel, min_score=min_score)

    if drop_duplicate:
        aligned = _dedup_aligned(aligned, comet_batch_size)

    _finalize_aligned(aligned, out, jsonl, **_ann_kwargs)


# ---------------------------------------------------------------------------
# expert translation batch
# ---------------------------------------------------------------------------


@app.command("parse-expert-translations")
def parse_expert_translation_batch(
    input_pdf: Annotated[
        Path, typer.Option("--input", "-i", exists=True, dir_okay=False, help="Expert translation PDF.")
    ],
    output_jsonl: Annotated[Path, typer.Option("--output", "-o", help="Aligned JSONL output.")],
    document_id: Annotated[
        Optional[str], typer.Option("--document-id", help="Override the PDF stem used as document ID.")
    ] = None,
) -> None:
    """Extract existing French–Mooré table row pairs and their QA metadata."""
    from moore_web.expert_translation_parser import parse_expert_translations, write_jsonl

    records = parse_expert_translations(input_pdf, doc_id=document_id)
    write_jsonl(records, output_jsonl)
    typer.echo(f"Wrote {len(records)} expert translation pairs → {output_jsonl}")


@app.command("export-reviewed")
def export_reviewed_units(
    db: Annotated[
        Path, typer.Option("--db", exists=True, dir_okay=False, help="Review-app SQLite DB.")
    ] = Path("data/review/reviews.sqlite3"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Snapshot directory.")] = Path(
        "data/reviewed"
    ),
    push: Annotated[bool, typer.Option("--push", help="Upload the snapshot to the HF dataset repo.")] = False,
    repo: Annotated[
        Optional[str], typer.Option("--repo", help="HF dataset repo (default: private reviewed repo).")
    ] = None,
    message: Annotated[
        Optional[str], typer.Option("--message", "-m", help="Commit message for --push.")
    ] = None,
) -> None:
    """Snapshot accepted review units to per-source JSONL, optionally pushing them to the Hub."""
    from moore_web.reviewed_export import DEFAULT_REPO, export_reviewed, push_reviewed

    summary = export_reviewed(db, output_dir)
    for source, counts in summary["sources"].items():
        skipped = f"  ({counts['skipped_units']} mismatched units skipped)" if counts["skipped_units"] else ""
        typer.echo(f"  {source}: {counts['rows']} rows from {counts['units']} units{skipped}")
    typer.echo(f"Wrote snapshot → {output_dir}  (latest review {summary['latest_review_at']})")
    if push:
        sha = push_reviewed(output_dir, repo or DEFAULT_REPO, message)
        typer.echo(
            f"Pushed {repo or DEFAULT_REPO}@{sha}; pin it as [reviewed].revision in fr_mos_sources.toml"
        )


# ---------------------------------------------------------------------------
# Mooré proverb app
# ---------------------------------------------------------------------------


@app.command("parse-moore-proverbs")
def parse_moore_proverb_app(
    input_dir: Annotated[
        Path, typer.Option("--input-dir", exists=True, file_okay=False, help="Archived proverb app directory.")
    ],
    output_jsonl: Annotated[Path, typer.Option("--output", "-o", help="Proverb pairs as JSONL.")],
) -> None:
    """Pair Mooré proverbs with their French renderings from the archived app."""
    from moore_web.moore_proverbs_parser import parse_moore_proverbs, write_jsonl

    records = parse_moore_proverbs(input_dir)
    write_jsonl(records, output_jsonl)
    typer.echo(f"Wrote {len(records)} Mooré/French proverb pairs → {output_jsonl}")


@app.command("parse-moore-tales")
def parse_moore_tale_app(
    input_dir: Annotated[
        Path, typer.Option("--input-dir", exists=True, file_okay=False, help="Archived tales app directory.")
    ],
    output_jsonl: Annotated[Path, typer.Option("--output", "-o", help="One record per tale, as JSONL.")],
) -> None:
    """Pair Mooré tales with their French translations from the archived app (Contes volume 5)."""
    from moore_web.moore_tales_parser import parse_moore_tales, write_jsonl

    tales = parse_moore_tales(input_dir)
    write_jsonl(tales, output_jsonl)
    typer.echo(f"Wrote {len(tales)} Mooré/French tales → {output_jsonl}")


# ---------------------------------------------------------------------------
# abcBurkina coépouses tale
# ---------------------------------------------------------------------------


@app.command("segment-abc-coepouses")
def segment_abc_coepouses_tale(
    fr_input: Annotated[Path, typer.Option("--fr-input", exists=True, dir_okay=False)],
    mo_input: Annotated[Path, typer.Option("--mo-input", exists=True, dir_okay=False)],
    output_jsonl: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Pair French and Mooré story beats in the archived coépouses tale."""
    from moore_web.abc_coepouses_parser import parse_abc_coepouses, write_jsonl

    units = parse_abc_coepouses(fr_input, mo_input)
    write_jsonl(units, output_jsonl)
    typer.echo(f"Wrote {len(units)} paired tale units → {output_jsonl}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
