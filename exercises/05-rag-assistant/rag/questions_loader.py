import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class EvalQuestion(BaseModel):
    """Represents test question schema for the rag system."""
    question: str = Field(description="The question to ask the rag system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal, unanswerable etc.)")
    answerable: bool = Field(description="Whether or not the question should be answerable")


def load_questions(path: Path) -> list[EvalQuestion]:
    """Loads and validates test cases from JSON (JSONL) file.

    Iterates through each line of the specified file, parses it into JSON, and validates
    it against the EvalQuestion Pydantic schema to ensure data integrity before evaluation.

    Args:
        path (Path): The file path to the tests JSONL dataset.

    Returns:
        list[EvalQuestion]: A list of validated EvalQuestion model instances.

    Raises:
        ValueError: If a line contains invalid JSON syntax or fails Pydantic schema validation,
            including the line number and detailed error context.
    """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: not valid JSON — {e}") from e
            try:
                questions.append(EvalQuestion(**raw))
            except ValidationError as e:
                raise ValueError(f"{path}:{line_num}: doesn't match EvalQuestion schema:\n{e}") from e
    return questions

