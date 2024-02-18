import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings

from constants import EMBEDDING_MODEL

model_name = EMBEDDING_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class MyEmbeddingFunction(EmbeddingFunction[Documents]):

    def __call__(self, input: Documents) -> Embeddings:
        embeddings_list = []
        
        for text in input:
            tokens = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            embeddings_list.append(embeddings)
        
        return embeddings_list