# scripts/agent_tools.py

from typing import ClassVar, List, Dict
import re
import os
import xml.etree.ElementTree as ET

from pydantic import PrivateAttr
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class DocRetriever(BaseTool):
    name:        ClassVar[str] = "doc_retriever"
    description: ClassVar[str] = "Retrieve top-k relevant Solr docs/issues given a query"

    _client:     PersistentClient  = PrivateAttr()
    _collection: any               = PrivateAttr()

    def __init__(self, persist_dir: str, collection_name: str, embed_model: str):
        super().__init__()
        self._client = PersistentClient(
            path=persist_dir,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=emb_fn
        )

    def _run(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self._collection._embedding_function([query])[0]
        res = self._collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]


class LogSearcher(BaseTool):
    name:        ClassVar[str] = "log_searcher"
    description: ClassVar[str] = "Search Solr log files for a regex pattern"

    log_dir: str

    def __init__(self, log_dir: str):
        super().__init__(log_dir=log_dir)
        self.log_dir = log_dir

    def _run(self, pattern: str, time_window: Dict[str, str] = None) -> List[Dict]:
        regex = re.compile(pattern)
        results = []
        for fname in os.listdir(self.log_dir):
            path = os.path.join(self.log_dir, fname)
            with open(path) as f:
                for line in f:
                    if regex.search(line):
                        results.append({"line": line.strip()})
        return results


class ConfigValidator(BaseTool):
    name:        ClassVar[str] = "config_validator"
    description: ClassVar[str] = "Validate Solr XML config and report errors/warnings"

    def _run(self, config_path: str) -> Dict:
        errors, warnings = [], []
        try:
            tree = ET.parse(config_path)
            root = tree.getroot()
            if root.find(".//schemaFactory") is None:
                errors.append("Missing <schemaFactory> element")
        except ET.ParseError as e:
            errors.append(f"XML parse error: {e}")
        return {"errors": errors, "warnings": warnings}


class Summarizer(BaseTool):
    name:        ClassVar[str] = "summarizer"
    description: ClassVar[str] = "Summarize long text into a brief summary"

    _chain: LLMChain = PrivateAttr()

    def __init__(self, llm):
        super().__init__()
        template = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following for a Solr engineer:\n\n{text}\n\nSummary:"
        )
        self._chain = LLMChain(llm=llm, prompt=template)

    def _run(self, text: str) -> str:
        return self._chain.run(text=text)

