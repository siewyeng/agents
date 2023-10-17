from copy import deepcopy
from csv import writer
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document


def _get_seconds_passed(time: datetime, ref_time: datetime | str) -> float:
    """Get the seconds passed between two datetime objects."""
    if isinstance(ref_time, str):
        ref_time = datetime.fromisoformat(ref_time)
    return (time - ref_time).total_seconds()


class ModTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    mem_file: str

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = str(current_time)
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = str(current_time)
            self.addToCSV(
                doc.metadata["created_at"],
                doc.metadata["last_accessed_at"],
                doc.page_content,
                doc.metadata["importance"],
            )
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime,
    ) -> float:
        """Return the combined score for a document."""
        seconds_passed = _get_seconds_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** seconds_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += int(float(document.metadata[key]))
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def addToCSV(self, created_at, last_accessed_at, observations, importance):
        List = [created_at, last_accessed_at, observations, importance]

        with open(self.mem_file, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
