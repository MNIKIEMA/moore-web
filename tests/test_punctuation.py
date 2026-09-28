import pytest

from moore_web.punctuation import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Embedded quote: the full stop belongs to the running sentence and stays outside.
        (
            "Rẽ daar tɛka, wãamb yʋʋr lebga «a pa tagsde».",
            "Rẽ daar tɛka, wãamb yʋʋr lebga « a pa tagsde ».",
        ),
        # Last piece of a split quote: the lone » goes.
        (
            "De cette manière seulement, l'hyène pourra faire son repas impunément "
            "et continuera vivre en bonne santé».",
            "De cette manière seulement, l'hyène pourra faire son repas impunément "
            "et continuera vivre en bonne santé.",
        ),
        # First piece of a split quote: the lone « goes.
        ("« Ah, voilà mon vieux qui vient !", "Ah, voilà mon vieux qui vient !"),
        (
            "Gomtɩʋʋg yeela ninsaala : « M pʋʋsd-f-la bark wʋsg f sẽn yiis-m yel-to-kãngã pʋgẽ wã.",
            "Gomtɩʋʋg yeela ninsaala : M pʋʋsd-f-la bark wʋsg f sẽn yiis-m yel-to-kãngã pʋgẽ wã.",
        ),
        # Whole quote on the line: its ! moves inside.
        (
            "« Ah, c’est les œufs de ma femme Poko. C'est elle qui les a trouvés la première "
            "et elle m'a dit les avoir vus »!",
            "« Ah, c’est les œufs de ma femme Poko. C'est elle qui les a trouvés la première "
            "et elle m'a dit les avoir vus ! »",
        ),
        ("« Mais quoi? Toi aussi ? Et tu vis ici » ?", "« Mais quoi ? Toi aussi ? Et tu vis ici ? »"),
        ("« Ayo, katre », reoog rabaamã sẽn yeele.", "« Ayo, katre », reoog rabaamã sẽn yeele."),
        ("Il dit : « Je suis arrivé. ».", "Il dit : « Je suis arrivé. »"),
        ("« C’est le chat».", "« C’est le chat. »"),
        # Lone » after the quote's own punctuation, then a second mark.
        ("La perdrix est morte! »", "La perdrix est morte !"),
        ("Koadeng kiime ! »!", "Koadeng kiime !"),
        # Link boundaries.
        ("il présente la gourde au vieux:", "Il présente la gourde au vieux."),
        ("en voulait une deuxième,", "En voulait une deuxième."),
        (
            "« Mam waa n na n bõos-f lame tɩ f põng m zugã »",
            "« Mam waa n na n bõos-f lame tɩ f põng m zugã. »",
        ),
        ("Il prend la gourde et la présente au vieux .", "Il prend la gourde et la présente au vieux."),
        ("si le lièvre est ,vraiment le plus malin", "Si le lièvre est, vraiment le plus malin."),
        ("Il a payé 1,5 franc.", "Il a payé 1,5 franc."),
        # Straight quotes are left alone.
        ('Il demanderait : "Hé, lièvre !"', 'Il demanderait : "Hé, lièvre !"'),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_titles_end_bare():
    assert normalize("Yir baag ne weoogẽ baaga.", is_title=True) == "Yir baag ne weoogẽ baaga"
    assert normalize("Plus intelligent qu’un autre?", is_title=True) == "Plus intelligent qu’un autre ?"


def test_times_keep_their_colon():
    assert normalize("Il est parti à 10:30.") == "Il est parti à 10:30."
