from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Document:
    content: str
    metadata: dict

@dataclass_json
@dataclass
class ToolCall:
    name: str
    query: str
    documents: List[Document]

@dataclass_json
@dataclass
class LLMAnswer:
    answer: str

@dataclass
class RAGResult:
    question: str
    answer: str
    context: List[str] = field(default_factory=list)
    