from typing import List

from langchain.document_loaders import CSVLoader
from langchain.schema import Document


def load_documents(mem_file: str = "./memory/memory.csv") -> List[Document]:
    """ "Load memory (history) from CSV "

    Returns
    -------
    list
        A list of documents where each row of the CSV is a document
    """
    loader = CSVLoader(
        mem_file, metadata_columns=["last_accessed_at", "created_at", "importance"]
    )
    docs = loader.load()
    return docs
