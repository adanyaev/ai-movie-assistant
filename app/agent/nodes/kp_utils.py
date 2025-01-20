import requests
import wikipedia
wikipedia.set_lang("ru")

from app.core.config import settings


BASE_URL = "https://api.kinopoisk.dev/v1.4"
MOVIE_SEARCH_URL = BASE_URL + "/movie"
MOVIE_SEARCH_BY_NAME_URL = BASE_URL + "/movie/search"
PERSON_SEARCH_BY_NAME_URL = BASE_URL + "/person/search"
REVIEW_SEARCH_URL = BASE_URL + "/review"


HEADERS = {
            "accept": "application/json",
            "X-API-KEY": settings.KP_API_KEY
        }


def transform_movie_data(movie_json: dict) -> str:

    name = movie_json.get("name", "Unknown Title")
    year = movie_json.get("year", "Unknown Year")
    description = movie_json.get("description", "No description available.")
    movie_type = movie_json.get("type", "Unknown type")
    is_series = movie_json.get("isSeries", False)
    movie_length = movie_json.get("movieLength", "Unknown length")
    series_length = movie_json.get("seriesLength", "N/A")
    total_series_length = movie_json.get("totalSeriesLength", "N/A")
    mpaa_rating = movie_json.get("ratingMpaa", "No MPAA rating")
    genres = ", ".join([genre["name"] for genre in movie_json.get("genres", [])])
    countries = ", ".join([country["name"] for country in movie_json.get("countries", [])])

    rating_data = movie_json.get("rating", {})
    kp_rating = rating_data.get("kp", "N/A")
    imdb_rating = rating_data.get("imdb", "N/A")
    critics_rating = rating_data.get("filmCritics", "N/A")
    russian_critics_rating = rating_data.get("russianFilmCritics", "N/A")
    await_rating = rating_data.get("await", "N/A")

    votes_data = movie_json.get("votes", {})
    kp_votes = votes_data.get("kp", "N/A")
    imdb_votes = votes_data.get("imdb", "N/A")
    critics_votes = votes_data.get("filmCritics", "N/A")

    if is_series:
        length_info = f"Episode Length: {series_length} minutes | Total Series Length: {total_series_length if total_series_length != 'N/A' else 'N/A'}"
    else:
        length_info = f"Duration: {movie_length} minutes"

    output = f"""
    Title: {name}
    Year: {year}
    Type: {movie_type.capitalize()}
    Country: {countries}
    Genres: {genres}
    {length_info}
    MPAA Rating: {mpaa_rating}

    Description: {description}

    Ratings:
    - Kinopoisk: {kp_rating} ({kp_votes} votes)
    - IMDb: {imdb_rating} ({imdb_votes} votes)
    - Film Critics: {critics_rating} ({critics_votes} votes)
    - Russian Film Critics: {russian_critics_rating} / 100
    - Awaiting Audience Rating: {await_rating}
    """

    return output.strip()

WIKI_SEARCH_KEYWORDS = (
        "актёр",
        "актер",
        "актриса",
        "режиссер",
        "режиссёр",
        "сценарист",
        "продюсер",
        "оператор",
        "композитор",
        "художник-постановщик",
        "гримёр",
        "костюмер",
        "звукорежиссёр",
        "монтажёр",
        "каскадёр",
        "хореограф",
        "дубляж",
        "театр",
        "кино",
        "мульт",
        "аниме",
        "сериал",
    )

def get_person_info_from_wiki(name: str, return_summary: bool = False) -> str | None:

    search_res = wikipedia.search(name, results=3)
    for r in search_res:
        page = wikipedia.page(r)
        for kw in WIKI_SEARCH_KEYWORDS:
            if page.summary.find(kw) != -1:
                break
        else:
            continue
        break
    else:
        return None  # Персона не найдена или не относится к киноиндустрии
    
    if return_summary:
        return page.summary
    return page.content


DEFAULT_SEARCH_PARAMS = {
    "page": 1,
    "limit": 5,
    "type": ["movie", "tv-series"],
    "sortField": "rating.kp",
    "sortType": "-1",
    "votes.kp": "10000-99999999999",
    #"persons.enProfession": ["actor", "director"],
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


GENRE_NAMES = (
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
    "реальное тв",
    "семейный",
    "спорт",
    "ток-шоу",
    "триллер",
    "ужасы",
    "фантастика",
    "фильм-нуар",
    "фэнтези",
    "церемония",
)

ITEM_TYPES = ("animated-series", "anime", "cartoon", "movie", "tv-series")


class InferKpId:

    @staticmethod
    def movie(item_name: str) -> int | None:
        item_name = item_name.strip("\"\' *\n")
        params = {
            "page": 1,
            "limit": 1,
            "query": item_name,
        }
        api_response = requests.get(
            MOVIE_SEARCH_BY_NAME_URL,
            params=params,
            headers=HEADERS,
        )
        if not api_response.ok:
            return None
        data_json = api_response.json()
        if not data_json["docs"]:
            return None
        
        return data_json["docs"][0]["id"]
    
    @staticmethod
    def person(item_name: str) -> int | None:
        item_name = item_name.strip("\"\' *\n")
        params = {
            "page": 1,
            "limit": 1,
            "query": item_name,
        }
        api_response = requests.get(
            PERSON_SEARCH_BY_NAME_URL,
            params=params,
            headers=HEADERS,
        )
        if not api_response.ok:
            return None
        data_json = api_response.json()
        if not data_json["docs"]:
            return None
        return data_json["docs"][0]["id"]
    
    @staticmethod
    def genre(item_name: str) -> str | None:
        try:
            idx = GENRE_NAMES.index(item_name.strip("\"\' *\n").lower())
        except ValueError:
            return None
        return idx
