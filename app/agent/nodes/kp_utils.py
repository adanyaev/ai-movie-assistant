from typing import Annotated, Literal, TypedDict


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


default_search_params = {
    "page": 1,
    "limit": 5,
    "type": ["movie", "tv-series"],
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


genre_names = (
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
)

all_item_types = ("animated-series", "anime", "cartoon", "movie", "tv-series")
