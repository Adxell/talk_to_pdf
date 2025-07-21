from pydantic import BaseModel

class RagInput(BaseModel):
    question: str