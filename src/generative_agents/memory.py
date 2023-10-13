import logging
import re
from datetime import datetime
from typing import List

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_experimental.generative_agents.memory import GenerativeAgentMemory

logger = logging.getLogger(__name__)


class StemssGenerativeAgentMemory(GenerativeAgentMemory):
    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        template = (
            "On the scale of 1 to 10, where 1 is not important at all"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely important (e.g., a break up, college"
            + " acceptance), rate the importance of the"
            + " following piece of memory. You must respond with a single integer."
            "\nMemory: {memory_content}"
            "\nRating: "
        )
        prompt = PromptTemplate.from_template(template)

        score = self.chain(prompt).run(memory_content=memory_content).strip()

        logger.warning(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * self.importance_weight
        else:
            return 0.0

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = datetime.fromisoformat(mem.metadata["created_at"]).strftime(
                "%B %d, %Y, %I:%M %p"
            )
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])
