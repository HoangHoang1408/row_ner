from typing import List, Union

from huggingface_hub import InferenceClient


class TEIClient:
    def __init__(self, server_url: str = "http://localhost:8080/embed"):
        self.client = InferenceClient()
        self.server_url = server_url

    def embed(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        return self.client.feature_extraction(inputs, model=self.server_url)
