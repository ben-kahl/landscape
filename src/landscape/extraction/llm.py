import ollama

from landscape.config import settings
from landscape.extraction.schema import Extraction

_SYSTEM_PROMPT = (
    "You are a precise knowledge-graph extractor. Given a passage of text, extract:\n"
    "1. Named entities with their types"
    " (PERSON, ORGANIZATION, PROJECT, TECHNOLOGY, LOCATION, OTHER).\n"
    "2. Relationships between those entities as (subject, relation_type, object) triples.\n"
    "\n"
    "CRITICAL rules:\n"
    "- Extract IMPLICIT relationships. A nominalized verb like 'approval' or 'migration'\n"
    "  still encodes a relationship: 'Sarah approved the PostgreSQL migration' means\n"
    "  Sarah -[APPROVED]-> PostgreSQL. The object of the approval is PostgreSQL, not\n"
    "  the migration event itself.\n"
    "- Prepositional phrases encode membership/affiliation:\n"
    "  'Alice leads Project Atlas at Acme Corp' means Alice -[WORKS_FOR]-> Acme Corp\n"
    "  AND Project Atlas -[BELONGS_TO]-> Acme Corp.\n"
    "- When the same entity appears under multiple surface forms (e.g. 'Project Atlas'\n"
    "  and 'Atlas'), list the canonical name once and put alternate forms in aliases.\n"
    "- Include only relationships between entities you have identified.\n"
    "- Use SCREAMING_SNAKE_CASE for relation_type (APPROVED, LEADS, WORKS_FOR, etc.).\n"
    "- Return strict JSON matching the schema. No prose, no markdown fences.\n"
    "\n"
    "--- EXAMPLE 1 ---\n"
    'Input: "Sarah approved the PostgreSQL migration."\n'
    "Reasoning: 'the PostgreSQL migration' is a nominalized event. The subject is Sarah,\n"
    "the action is APPROVED, and the direct object technology is PostgreSQL.\n"
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Sarah", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "PostgreSQL", "type": "TECHNOLOGY", "confidence": 0.95, "aliases": []}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Sarah", "object": "PostgreSQL",'
    ' "relation_type": "APPROVED", "confidence": 0.9}\n'
    "  ]\n"
    "}\n"
    "\n"
    "--- EXAMPLE 2 ---\n"
    'Input: "Alice leads Project Atlas at Acme Corp."\n'
    "Reasoning: 'leads' = Alice LEADS Project Atlas. 'at Acme Corp' = Alice WORKS_FOR\n"
    "Acme Corp AND Project Atlas BELONGS_TO Acme Corp.\n"
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Alice", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Project Atlas", "type": "PROJECT", "confidence": 0.95,'
    ' "aliases": ["Atlas"]},\n'
    '    {"name": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.95, "aliases": []}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Alice", "object": "Project Atlas",'
    ' "relation_type": "LEADS", "confidence": 0.9},\n'
    '    {"subject": "Alice", "object": "Acme Corp",'
    ' "relation_type": "WORKS_FOR", "confidence": 0.85},\n'
    '    {"subject": "Project Atlas", "object": "Acme Corp",'
    ' "relation_type": "BELONGS_TO", "confidence": 0.85}\n'
    "  ]\n"
    "}\n"
    "\n"
    "--- EXAMPLE 3 ---\n"
    'Input: "Project Atlas uses PostgreSQL for storage. Sarah is on the Platform Team."\n'
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Project Atlas", "type": "PROJECT", "confidence": 0.95,'
    ' "aliases": ["Atlas"]},\n'
    '    {"name": "PostgreSQL", "type": "TECHNOLOGY", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Sarah", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Platform Team", "type": "ORGANIZATION", "confidence": 0.9,'
    ' "aliases": []}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Project Atlas", "object": "PostgreSQL",'
    ' "relation_type": "USES", "confidence": 0.95},\n'
    '    {"subject": "Sarah", "object": "Platform Team",'
    ' "relation_type": "MEMBER_OF", "confidence": 0.9}\n'
    "  ]\n"
    "}\n"
    "\n"
    "Now extract from the following text:"
)


def extract(text: str) -> Extraction:
    client = ollama.Client(host=settings.ollama_url)
    response = client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": f"{_SYSTEM_PROMPT}\n\n{text}"},
        ],
        format=Extraction.model_json_schema(),
    )
    return Extraction.model_validate_json(response.message.content)
