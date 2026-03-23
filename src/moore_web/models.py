import msgspec


class Example(msgspec.Struct):
    moore: str
    french: str
    english: str | None = None


class Sense(msgspec.Struct):
    id: str
    french: str
    english: str
    examples: list[Example] = []
    category: str | None = None
    scientific_name: str | None = None
    synonym: str | None = None
    antonym: str | None = None
    pos: str | None = None


class DictionaryEntry(msgspec.Struct):
    lemma: str
    ipa: str
    pos: str
    senses: list[Sense]
    variants: dict[str, list[str]] | None = None
