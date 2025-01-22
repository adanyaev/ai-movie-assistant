import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from app.core.config import settings

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.ENCODER_MODEL_NAME,
)

client = chromadb.HttpClient(host=settings.INDEX_DB_HOST, port=settings.INDEX_DB_PORT)


collection = client.create_collection(
    name=settings.MOVIES_COLLECTION_NAME,
    embedding_function=openai_ef,
    metadata={
        "description": "Movies DB",
        "hnsw:space": "cosine",
    },
    get_or_create=True,
)


def populate_index_db() -> None:
    import copy
    import requests
    from app.agent.nodes import kp_utils


    params = copy.deepcopy(kp_utils.DEFAULT_SEARCH_PARAMS)
    params["limit"] = 50
    params["lists"] = ["top250"]

    api_response = requests.get(kp_utils.MOVIE_SEARCH_URL, params=params, headers=kp_utils.HEADERS)
    if not api_response.ok:
        print("Произошла ошибка при обращении к API")

    docs = api_response.json()["docs"]
    for i, doc in enumerate(docs):
        collection.add(
            documents=doc["description"],
            metadatas={
                "movie_name": doc.get("name", "Unknown Title"),
                "movie_data": kp_utils.transform_movie_data(doc),
            },
            ids=str(i),
        )
    print("Index DB population complete")


def test_index_db() -> None:
    query = (
        "Фильм про бухгалтера обвинённого в убийстве собственной жены и её любовника"
    )
    response = collection.query(query_texts=query, n_results=1)
    print(response)


def drop_index_db() -> None:
    client.delete_collection(settings.MOVIES_COLLECTION_NAME)
    print("Index DB drop complete")
