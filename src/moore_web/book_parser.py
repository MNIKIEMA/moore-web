import pymupdf


def parse_book(doc: pymupdf.Document):
    # TODO: Find alignment automatically @critical
    all_parts = []
    for i, page in enumerate(doc, start=1):  #type: ignore
        page_width = page.rect.width
        middle_x = page_width / 2
        moore_parts = []
        french_parts = []
        if i < 3:
            continue
        if i > 47:
            break
        blocks = page.get_text("blocks", sort = True)
        for b in blocks:
            x0, y0, x1, y1, text = b[:5]
            text = text.strip()

            if not text:
                continue
            if text.isdigit():
                continue

            if x0 < middle_x:
                moore_parts.append(text)
            else:
                french_parts.append(text)

        all_parts.append({"moore": moore_parts, "french": french_parts, "page": i})

    return all_parts
