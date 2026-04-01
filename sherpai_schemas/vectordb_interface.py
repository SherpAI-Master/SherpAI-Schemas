"""Handling mivlus setup and embeddings."""

from pathlib import Path

import pandas as pd
from pymilvus import DataType, MilvusClient

from .llm_interface import batch_vectorization


def setup_milvus(client: MilvusClient, collection_name: str) -> None:
    """Create a Milvus collection with the given name, if it does not yet exist.

    The collection has the following fields:
    - id: a string (VARCHAR) of at most 20 characters, used as a primary key
    - string_data: a string (VARCHAR) of at most 300 characters
    - vector: a float vector of length 384 (FLOAT_VECTOR)
    - json_data: a JSON object (JSON)

    The collection is created with auto_id=False and enable_dynamic_field=False

    :param client: MilvusClient object
    :param collection_name: the name of the collection to create
    """
    existing_collections = client.list_collections()
    if collection_name in existing_collections:
        return

    # Schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=20)
    schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=300)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)
    schema.add_field(field_name="json_data", datatype=DataType.JSON)

    # Collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )


def query_db(
    search_df: pd.DataFrame,
    collection_name: str,
    milvus_client: MilvusClient,
    batch_size: int = 512,
    limit: int = 3,
) -> list[list[dict]]:
    """Query the Milvus database with the given a DataFrame of elements to be deduplicated.

    :param search_strings (list[float]): A list of vectors to search the database with.
    :param collection_name (str): The name of the collection to search in.
    :param milvus_path (Path): The path of the milvus DB
    :param batch_size: Batch size of search in vector DB
    :type batch_size: int
    :param limit: Limit of responses per search
    :type limit: int
    :return: MilvusResponse: The response from the Milvus search query.
    """
    print(milvus_client.list_collections())
    milvus_client.load_collection(collection_name=collection_name)

    search_strings: pd.Series = search_df.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)
    embedded_search_strings = batch_vectorization(search_strings)

    results = []
    for i in range(0, len(embedded_search_strings), batch_size):
        search_batch = embedded_search_strings[i : i + batch_size]

        response = milvus_client.search(
            collection_name=collection_name,
            anns_field="vector",
            data=search_batch,
            limit=limit,
            search_params={"metric_type": "COSINE"},
            output_fields=["json_data"],
        )
        results.extend(response)

    milvus_client.release_collection(collection_name=collection_name)

    return results


def _extend_row(row: pd.Series) -> pd.Series:
    """Create document to be stringified and json object for VectorDB.

    :param row: Current row of df
    :type row: pd.Series
    :return: Altered row
    :rtype: pd.Series
    """
    row["doc"] = " ".join(row.dropna().astype(str))
    row["json_data"] = row.drop("doc").to_json()
    return row


def _create_indexing(client: MilvusClient, collection_name: str) -> None:
    """Create an index on the given collection for the vector field.

    The index is an IVF_SQ8 index with a cosine metric and 128 clusters.

    :param client: a MilvusClient object
    :param collection_name: the name of the collection to index
    """
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",  # IVF_SQ8
        index_name="index_file",
        params={"nlist": 128},
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        sync=True,
    )


def vectorize_data(
    client: MilvusClient,
    data: Path | str,
    batch_size: int = 512,
    collection_name: str = "main",
) -> None:
    """Vectorize data and insert into vectorDB collection.

    :param data: JSONL file with data to be vectorized
    :type data: Path | str
    :param embedding_model: Embedding model (from HF)
    :type embedding_model: str
    :param batch_size: batch size of vectorization
    :type batch_size: int
    :param collection_name: MilvusDB collection name
    :type collection_name: str
    """
    setup_milvus(client, collection_name)

    # Read in data
    df = pd.read_json(data, lines=True)
    df = df.apply(_extend_row, axis=1)

    # Vectorize stuff
    df["vector"] = batch_vectorization(df["doc"])

    # Get into schema {id, doc, embedding, json_data}
    data = df[["hybrid", "doc", "vector", "json_data"]].rename(columns={"hybrid": "id"})
    data = data.to_dict("records")

    # Add to collection
    for i in range(0, len(df), batch_size):
        batch = data[i : i + batch_size]
        client.insert(collection_name=collection_name, data=batch)

    _create_indexing(client, collection_name)
