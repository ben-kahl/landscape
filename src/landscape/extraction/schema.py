from pydantic import BaseModel


class ExtractedEntity(BaseModel):
    name: str
    type: str
    confidence: float
    aliases: list[str] = []


class ExtractedRelation(BaseModel):
    subject: str
    object: str
    relation_type: str
    confidence: float


class Extraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
