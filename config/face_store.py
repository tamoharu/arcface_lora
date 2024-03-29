from config.type import Embedding

source_embedding: Embedding = None

def reset_face_store():
    global source_embedding
    source_embedding = None 