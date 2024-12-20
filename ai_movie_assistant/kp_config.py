from typing import Annotated, Literal, TypedDict


default_search_params = {
    "page": 1,
    "limit": 5,
    "sortField": "rating.kp",
    "sortType": "-1",
    "votes.kp": "10000-99999999999",
    "selectFields": [
        "id",
        "type",
        "name",
        "rating",
        "description",
        "votes",
        "year",
        # "poster",
        "genres",
        "countries",
        # "typeNumber",
        # "alternativeName",
        # "backdrop",
        "enName",
        "movieLength",
        # "names",
        # "status",
        "ratingMpaa",
        "shortDescription",
        # "ticketsOnSale",
        # "ageRating",
        # "logo",
        "releaseYears",
        # "top10",
        # "top250",
        "isSeries",
        "seriesLength",
        "totalSeriesLength",
    ],
}


genre_names = Literal[
    "аниме",
    "биография",
    "боевик",
    "вестерн",
    "военный",
    "детектив",
    "детский",
    "для взрослых",
    "документальный",
    "драма",
    "игра",
    "история",
    "комедия",
    "концерт",
    "короткометражка",
    "криминал",
    "мелодрама",
    "музыка",
    "мультфильм",
    "мюзикл",
    "новости",
    "приключения",
    "реальное ТВ",
    "семейный",
    "спорт",
    "ток-шоу",
    "триллер",
    "ужасы",
    "фантастика",
    "фильм-нуар",
    "фэнтези",
    "церемония",
]

item_types = Literal["animated-series", "anime", "cartoon", "movie", "tv-series"]
