from typing import List, Union

from huggingface_hub import InferenceClient


class TEIClient:
    def __init__(self, server_url: str = "http://localhost:8080/embed"):
        self.client = InferenceClient()
        self.server_url = server_url

    def embed(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for a single string or a list of strings.
        Args:
            inputs (str or List[str]): The input text(s) to embed.
        Returns:
            List[List[float]]: Embedding(s) for the input(s).
        """
        return self.client.feature_extraction(inputs, model=self.server_url)
