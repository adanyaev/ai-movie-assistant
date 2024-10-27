from typing import Annotated, Literal, TypedDict, Optional
import json
import os
import copy

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool

import kp_config


API_KEY = os.environ.get("KP_API_KEY")

BASE_URL = " https://api.kinopoisk.dev/v1.4/movie"

headers = {
        "X-API-KEY": API_KEY
    }

# @tool()
# def get_random_movie():
#     """Call to get info about random movie."""

#     response = json.dumps(api_funcs.get_random_movie())
#     return response

def filter_search_results_fields(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k in kp_config.default_search_params['selectFields']}


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
    Title: {name} ({year})
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
      - Russian Film Critics: {russian_critics_rating}
      - Awaiting Audience Rating: {await_rating}
    """
    
    return output.strip()


@tool()
async def get_movie_by_title(
    title: Annotated[str, "movie title"]
    ):
    """Call to get info about a movie by title. Results are sorted by descending relevance."""

    url = f"{BASE_URL}/search"
    
    params = {
        "page": 1,
        "limit": 1,
        "query": title,
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    # if response.status_code == 200:
    #     response = [filter_search_results_fields(i) for i in response.json()['docs']]
    #     return json.dumps(response)
    if response.status_code == 200:
        response = [transform_movie_data(filter_search_results_fields(i)) for i in response.json()['docs']]
        response = "\n".join(response)
        return response
    else:
        return json.dumps({"error": response.status_code, "message": response.text})


class FiltersInput(BaseModel):
    item_types: list[kp_config.item_types] = Field(description="Search item types to include, e. g. movie or series")
    genres_include: Optional[list[kp_config.genre_names]] = Field(description="Genres names to include in search results", default=[])
    genres_exclude: Optional[list[kp_config.genre_names]] = Field(description="Genres names to exclude from search results", default=[])
    


@tool(args_schema=FiltersInput)
async def search_movies_with_filters(item_types, genres_include=[], genres_exclude=[]):
    """Call to search for a movies with filters. Results are sorted by descending rating."""

    url = f"{BASE_URL}"
    params = copy.deepcopy(kp_config.default_search_params)
    params.update({
        "type": item_types,
    })
    genres = []
    if genres_include:
        genres.extend(genres_include)
    if genres_exclude:
        genres.extend(list(map(lambda x: f"!{x}", genres_exclude)))
    if genres:
        params["genres.name"] = genres
    response = requests.get(url, headers=headers, params=params)
    # if response.status_code == 200:
    #     return json.dumps(response.json()['docs'])
    if response.status_code == 200:
        response = [transform_movie_data(i) for i in response.json()['docs']]
        response = "\n".join(response)
        return response
    else:
        return json.dumps({"error": response.status_code, "message": response.text})
