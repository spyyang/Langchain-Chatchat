from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import dashscope
from http import HTTPStatus
from configs.model_config import embedding_model_dict

from typing import Any, List, Generator

DASHSCOPE_MAX_BATCH_SIZE = 25


class MyEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(text, normalize_embeddings=True)
        return embedding.tolist()


class DashscopeEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def batched(self, texts: List[str], batch_size: int = DASHSCOPE_MAX_BATCH_SIZE) -> Generator[List, None, None]:
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = list()
        batch_counter = 0
        for batch in self.batched(texts):
            resp = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v2,
                input=batch,
                api_key=embedding_model_dict['dashscope'],
            )
            if resp.status_code == HTTPStatus.OK:
                for emb in resp.output['embeddings']:
                    result.append(emb.get('embedding'))
            batch_counter += len(batch)
        return result

    def embed_query(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v2,
            input=text,
            api_key=embedding_model_dict['dashscope'],
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output.get('embeddings')[0].get('embedding')
