import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings

from constants import EMBEDDING_MODEL

# Initialize the tokenizer and model from HuggingFace
model_name = EMBEDDING_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class MyEmbeddingFunction(EmbeddingFunction[Documents]):
    '''The embedding function for the database. The format of this class is compatible with ChromaDB.
    
    Functions:
        __call__: Embeds the input documents and returns the embeddings.
        
    For more information, see the ChromaDB documentation: https://docs.trychroma.com/embeddings
    '''

    def __call__(self, input: Documents) -> Embeddings:
        '''Embeds the input documents and returns the embeddings.'''
        
        # Initilize the list to store the embeddings
        embeddings_list = []
        
        # Loop through the list of text documents
        for text in input:
            tokens = tokenizer(text, return_tensors='pt')
            with torch.no_grad(): # Do not calculate gradients
                outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            embeddings_list.append(embeddings)
        
        return embeddings_list