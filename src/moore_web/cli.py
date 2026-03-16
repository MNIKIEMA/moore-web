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
from typing import Annotated, Optional

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
    conseils = "conseils"


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


# TODO: replace Kadé by Poko and Katiu, Atega too


def _load_kade_book(path: Path):
    from moore_web.book_parser_facilitateur import parse_book_from_json

    return parse_book_from_json(str(path))


def _dedup_aligned(aligned):
    """Deduplicate an AlignedCorpus using COMET-QE and return a new one."""
    from moore_web.dedup_aligned_comet import deduplicate_by_comet
    from moore_web.flatten import AlignedCorpus

    pairs = [{"fr": f, "mo": m, "score": s} for f, m, s in zip(aligned.french, aligned.moore, aligned.scores)]
    typer.echo("      Running COMET-QE deduplication…")
    pairs = deduplicate_by_comet(pairs)
    return AlignedCorpus(
        french=[p["fr"] for p in pairs],
        moore=[p["mo"] for p in pairs],
        scores=[p["score"] for p in pairs],
        source=aligned.source,
    )


def _parse_kade_file(input_path: Path, lang: KadeLang):
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
    else:
        sec_titles = SECTION_TITLES
        intro_titles = FRENCH_INTRO_SECTION_TITLES
        intro_sub_titles = FRENCH_INTRO_SUBSECTION_TITLES
        intro_sub_key = "Comment utiliser ce manuel"

    sec_patterns = [_re.compile(_re.escape(t), _re.IGNORECASE) for t in sec_titles]
    intro_patterns = [_re.compile(_re.escape(t), _re.IGNORECASE) for t in intro_titles]
    intro_sub_patterns = [
        _re.compile(r"(?:\d+\.\s+)?" + _re.escape(t), _re.IGNORECASE) for t in intro_sub_titles
    ]
    intro_sub_map = {intro_sub_key: (intro_sub_patterns, intro_sub_titles)}

    if input_path.suffix.lower() == ".pdf":
        from moore_web.pdf_extractor import extract_pdf_blocks

        text = extract_pdf_blocks(str(input_path))
    else:
        text = input_path.read_text(encoding="utf-8")

    return parse_with_chapters(
        text,
        sec_patterns,
        sec_titles,
        intro_section_regexes=intro_patterns,
        intro_section_titles=intro_titles,
        intro_subsection_map=intro_sub_map,
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
            from moore_web.lang_id import annotate_text_units

            typer.echo("Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {len(corpus)} entries → {out}")

    elif source == Source.simple:
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.simple_parser import parse_doc

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

    elif source == Source.simple:
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
            from moore_web.lang_id import annotate_text_units

            typer.echo("Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        parallel = flatten_news_entries(corpus, segment=segment)
        out = output or _default_output(input, "_parallel.json")

    elif source == Source.simple:
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.flatten import flatten_simple_parser
        from moore_web.simple_parser import parse_doc

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
        typer.Option("--min-score", min=0.0, max=1.0, help="Drop pairs below this cosine similarity."),
    ] = 0.0,
) -> None:
    """Align a ParallelText JSON using LASER embeddings + FastDTW.

    Example: moore-web align parallel.json -o aligned.json --min-score 0.6
    """
    from moore_web.align_corpus import align as _align
    from moore_web.flatten import ParallelText

    out = output or _default_output(input, "_aligned.json")

    parallel = ParallelText.from_json(input.read_bytes())
    typer.echo(f"Input: {len(parallel.french)} FR  {len(parallel.moore)} MO")

    aligned = _align(parallel, min_score=min_score)

    out.write_bytes(msgspec.json.encode(aligned))
    typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")


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
        typer.Option("--output", "-o", help="Output aligned JSON (default: derived from input)."),
    ] = None,
    segment: Annotated[
        bool,
        typer.Option("--segment/--no-segment", help="Sentence-segment each text block."),
    ] = True,
    min_score: Annotated[
        float,
        typer.Option("--min-score", min=0.0, max=1.0, help="Drop pairs below this cosine similarity."),
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
    drop_duplicate: Annotated[
        bool,
        typer.Option(
            "--drop-duplicate/--no-drop-duplicate",
            help="Deduplicate aligned pairs with COMET-QE, keeping highest score per group (not available for simple).",
        ),
    ] = False,
) -> None:
    """End-to-end pipeline: parse → flatten → align.

    [bold]sida:[/bold]    moore-web e2e -s sida -i book.pdf -o aligned.json
    [bold]kade:[/bold]    moore-web e2e -s kade --fr-input fr.pdf --mo-input mo.pdf -o aligned.json
    [bold]news:[/bold]    moore-web e2e -s news -i corpus.json -o aligned.json
    [bold]simple:[/bold]  moore-web e2e -s simple -i dict.pdf -o aligned.json
    """
    from moore_web.align_corpus import align as _align
    from moore_web.flatten import (
        flatten_facilitateur_pair,
        flatten_sida_book,
    )

    # ── parse + flatten ──────────────────────────────────────────────────────
    if source == Source.sida:
        if input is None:
            _err("--input is required for source 'sida'.")
            raise typer.Exit(1)
        from moore_web.book_parser import parse_pdf_to_json

        typer.echo(f"[1/3] Parsing SIDA book: {input}")
        chapters = parse_pdf_to_json(str(input))
        typer.echo(f"[2/3] Flattening {len(chapters)} chapters…")
        parallel = flatten_sida_book(chapters, segment=segment)
        out = output or _default_output(input, "_aligned.json")

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
        out = output or fr_input.with_name("kade_aligned.json")

    elif source == Source.news:
        if input is None:
            _err("--input is required for source 'news'.")
            raise typer.Exit(1)
        typer.echo(f"[1/3] Parsing news corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))

        if lang_id:
            from moore_web.lang_id import annotate_text_units

            typer.echo("      Running language ID…")
            corpus = annotate_text_units(corpus)

        from moore_web.flatten import AlignedCorpus, flatten_news_per_entry
        from moore_web.segment_news_data import segment_entries

        corpus = segment_entries(corpus)
        typer.echo("[2/3] Flattening…")
        article_parallels = flatten_news_per_entry(corpus, segment=segment)
        out = output or _default_output(input, "_aligned.json")
        typer.echo(f"      {len(article_parallels)} bilingual articles found.")

        typer.echo("[3/3] Aligning per article with LASER + FastDTW…")
        all_fr, all_mo, all_scores = [], [], []
        for url, dp in article_parallels:
            aligned_dp = _align(dp, min_score=min_score)
            all_fr.extend(aligned_dp.french)
            all_mo.extend(aligned_dp.moore)
            all_scores.extend(aligned_dp.scores)

        aligned = AlignedCorpus(
            french=all_fr,
            moore=all_mo,
            scores=all_scores,
            source="news",
        )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned)
        out.write_bytes(msgspec.json.encode(aligned))
        typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")
        return

    elif source == Source.simple:
        if input is None:
            _err("--input is required for source 'simple'.")
            raise typer.Exit(1)
        import pymupdf

        from moore_web.flatten import AlignedCorpus, flatten_simple_parser
        from moore_web.simple_parser import parse_doc

        typer.echo(f"[1/2] Parsing simple dictionary: {input}")
        with pymupdf.open(str(input)) as doc:
            pages = parse_doc(doc)
        typer.echo("[2/2] Flattening…")
        parallel = flatten_simple_parser(pages, include_examples=examples, include_entries=entries)
        out = output or _default_output(input, "_aligned.json")

        typer.echo(
            f"      FR: {len(parallel.french)}  MO: {len(parallel.moore)}  EN: {len(parallel.english)}"
        )
        # Data is pre-aligned by construction — no LASER needed.
        aligned = AlignedCorpus(
            french=parallel.french,
            moore=parallel.moore,
            english=parallel.english,
            scores=[1.0] * len(parallel.french),
            source=parallel.source,
        )
        out.write_bytes(msgspec.json.encode(aligned))
        typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")
        return

    elif source == Source.conseils:
        if input is None:
            _err("--input is required for source 'conseils'.")
            raise typer.Exit(1)
        from moore_web.flatten import AlignedCorpus, flatten_conseils

        typer.echo(f"[1/2] Flattening conseil-des-ministres corpus: {input}")
        corpus = json.loads(input.read_text(encoding="utf-8"))
        date_parallels = flatten_conseils(corpus, segment=segment)
        out = output or _default_output(input, "_aligned.json")
        typer.echo(f"      {len(date_parallels)} bilingual sessions found.")

        # Align each date independently, then concatenate.
        typer.echo("[2/2] Aligning per date with LASER + FastDTW…")
        all_fr, all_mo, all_scores = [], [], []
        for date, dp in date_parallels:
            typer.echo(f"      {date}: FR={len(dp.french)}  MO={len(dp.moore)}")
            aligned_dp = _align(dp, min_score=min_score)
            all_fr.extend(aligned_dp.french)
            all_mo.extend(aligned_dp.moore)
            all_scores.extend(aligned_dp.scores)

        aligned = AlignedCorpus(
            french=all_fr,
            moore=all_mo,
            scores=all_scores,
            source="conseils",
        )
        if drop_duplicate:
            aligned = _dedup_aligned(aligned)
        out.write_bytes(msgspec.json.encode(aligned))
        typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")
        return

    typer.echo(f"      FR: {len(parallel.french)} sentences  MO: {len(parallel.moore)} sentences")

    # ── align ────────────────────────────────────────────────────────────────
    typer.echo("[3/3] Aligning with LASER + FastDTW…")
    aligned = _align(parallel, min_score=min_score)

    if drop_duplicate:
        aligned = _dedup_aligned(aligned)

    out.write_bytes(msgspec.json.encode(aligned))
    typer.echo(f"Wrote {len(aligned.french)} aligned pairs → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
