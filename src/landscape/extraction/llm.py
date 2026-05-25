import calendar
import logging
from datetime import UTC, datetime
from typing import Literal

import ollama

from landscape.config import LLM_PROFILES, settings

_iso_log = logging.getLogger(__name__)


def _parse_iso_temporal(
    value: str | None,
    *,
    is_endpoint: Literal["from", "until"],
) -> str | None:
    """Parse a YYYY / YYYY-MM / YYYY-MM-DD / full ISO datetime string and
    return an ISO-8601 UTC string with period-convention resolution.

    For partial inputs:
      * is_endpoint='from'  → start of period (Jan 1 00:00:00 / 1st 00:00:00 / 00:00:00)
      * is_endpoint='until' → end of period   (Dec 31 23:59:59 / last day 23:59:59 / 23:59:59)

    Returns None on unparseable input (logs a warning) or on empty/None.
    Never raises.
    """
    if not value:
        return None
    s = value.strip()

    # YYYY
    if len(s) == 4 and s.isdigit():
        year = int(s)
        dt = (
            datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
            if is_endpoint == "from"
            else datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
        )
        return dt.isoformat()

    # YYYY-MM
    if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
        year, month = int(s[:4]), int(s[5:7])
        if is_endpoint == "from":
            dt = datetime(year, month, 1, 0, 0, 0, tzinfo=UTC)
        else:
            last_day = calendar.monthrange(year, month)[1]
            dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=UTC)
        return dt.isoformat()

    # YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            d = datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            _iso_log.warning("could not parse temporal value as YYYY-MM-DD: %r", value)
            return None
        if is_endpoint == "from":
            dt = d.replace(hour=0, minute=0, second=0, tzinfo=UTC)
        else:
            dt = d.replace(hour=23, minute=59, second=59, tzinfo=UTC)
        return dt.isoformat()

    # Full ISO datetime — datetime.fromisoformat handles most variants since 3.11
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        _iso_log.warning("could not parse temporal value as ISO datetime: %r", value)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat()
from landscape.extraction.schema import Extraction
from landscape.middleware.token_counter import increment_ollama_tokens
from landscape.observability.weave_tracing import record_token_usage, traced

_SYSTEM_PROMPT = (
    "You are a precise knowledge-graph extractor. Given a passage of text, extract:\n"
    "1. Named entities with their types (PERSON, ORGANIZATION, PROJECT, TECHNOLOGY,\n"
    "   LOCATION, CONCEPT, EVENT, DOCUMENT, ROLE, DATETIME, TASK).\n"
    "2. Relationships between those entities as (subject, relation_type, object)\n"
    "   triples, with optional `subtype` and optional numeric qualifiers:\n"
    "   `quantity_value`, `quantity_unit`, `quantity_kind`, and `time_scope`.\n"
    "\n"
    "CRITICAL rules:\n"
    "- Extract IMPLICIT relationships. A nominalized verb like 'approval' or 'migration'\n"
    "  still encodes a relationship: 'Sarah approved the PostgreSQL migration' means\n"
    "  Sarah -[APPROVED]-> PostgreSQL.\n"
    "- Prepositional phrases encode membership/affiliation:\n"
    "  'Alice leads Project Atlas at Acme Corp' means Alice -[WORKS_FOR]-> Acme Corp\n"
    "  AND Project Atlas -[BELONGS_TO]-> Acme Corp.\n"
    "- When the same entity appears under multiple surface forms (e.g. 'Project Atlas'\n"
    "  and 'Atlas'), list the canonical name once and put alternate forms in aliases.\n"
    "- Include only relationships between entities you have identified.\n"
    "- Use SCREAMING_SNAKE_CASE for relation_type.\n"
    "- CRITICAL — choose relation_type ONLY from this closed vocabulary:\n"
    "  WORKS_FOR, LEADS, MEMBER_OF, WORKS_ON, MAINTAINS, OWNS, REPORTS_TO,\n"
    "  APPROVED, USES, DEPENDS_ON, BELONGS_TO, LOCATED_IN, CREATED, RELATED_TO,\n"
    "  HAS_TITLE, HAS_PREFERENCE, HAS_ATTRIBUTE, FAMILY_OF, RECOMMENDED,\n"
    "  DISCUSSED, HAPPENED_ON, LIVES_IN.\n"
    "  Synonym hints:\n"
    "    'employed by'/'hired by'/'works at' → WORKS_FOR\n"
    "    'manages'/'heads' → LEADS\n"
    "    'part of a team' → MEMBER_OF\n"
    "    'is working on'/'is assigned to'/'contributing to' → WORKS_ON\n"
    "      (subject=Person/Team, object=Project/Task; use over LEADS when no ownership implied)\n"
    "    'maintains'/'keeps running'/'responsible for upkeep of' → MAINTAINS\n"
    "      (subject=Person/Team, object=System/Service/Codebase)\n"
    "    'owns'/'is the owner of'/'has ownership of' → OWNS\n"
    "      (asset/feature/codebase ownership; not org hierarchy — use BELONGS_TO for that)\n"
    "    'signed off on'/'authorized' → APPROVED\n"
    "    'uses as a tool or library'/'built on'/'runs on' → USES\n"
    "    'depends on'/'requires'/'cannot function without' → DEPENDS_ON\n"
    "      (hard runtime/build dependency; stronger than USES)\n"
    "    'subsidiary of'/'owned by an org' → BELONGS_TO\n"
    "    'based in' (org/office) → LOCATED_IN\n"
    "    'authored'/'built'/'founded' → CREATED\n"
    "    'is a senior engineer'/'promoted to director'/'title is X' → HAS_TITLE\n"
    "      (subject=Person, object=Organization, subtype=snake_case(title))\n"
    "    'favorite color is X'/'prefers X'/'likes X best' → HAS_PREFERENCE\n"
    "      (subtype=axis, e.g. favorite_color, favorite_food, dietary)\n"
    "    'height is 6ft'/'pronouns are they/them' → HAS_ATTRIBUTE\n"
    "      (subtype=attribute name, e.g. height, pronouns)\n"
    "    'daughter'/'uncle'/'married to'/'spouse of' → FAMILY_OF\n"
    "      (subtype=relation kind, e.g. daughter, uncle, spouse, parent)\n"
    "    'recommended X'/'suggested' → RECOMMENDED\n"
    "    'talked about'/'mentioned' → DISCUSSED\n"
    "    'happened on'/'occurred on'/'scheduled for' → HAPPENED_ON\n"
    "      (subject=Event, object=DateTime)\n"
    "    'lives in'/'resides in'/'home is' → LIVES_IN (personal residence only)\n"
    "  If nothing fits, use RELATED_TO. Do NOT invent new relation_types.\n"
    "- `subtype` is optional. When the relation_type collapses nuance (title, family\n"
    "  kind, preference axis), populate `subtype` with a short snake_case phrase that\n"
    "  preserves the specifics. Omit (null) otherwise.\n"
    "- When a relationship includes a count, duration, frequency, price, distance,\n"
    "  percentage, rating, or measurement, preserve it on the relation using:\n"
    "  `quantity_value`, `quantity_unit`, `quantity_kind`, and `time_scope`.\n"
    "  Examples: 10 hours → quantity_value=10, quantity_unit=hour,\n"
    "  quantity_kind=duration; three bikes → quantity_value=3,\n"
    "  quantity_unit=bike, quantity_kind=count. Keep relation_type in the closed\n"
    "  vocabulary; quantities are edge qualifiers, not new relation types.\n"
    "- NEGATION — when a sentence negates a relationship, set negated: true and keep\n"
    "  the same relation_type from the closed vocab. Do NOT invent a 'NOT_' relation_type.\n"
    "  Negation markers: 'does not', 'doesn't', 'isn't', 'no longer', 'never', 'stopped',\n"
    "  'has not', 'haven't', 'wasn't', 'won't'.\n"
    "  Examples:\n"
    "    'Alice does not work for Acme' → relation_type: WORKS_FOR, negated: true\n"
    "    'Project X no longer depends on Redis' → relation_type: DEPENDS_ON, negated: true\n"
    "    'We do not recommend DynamoDB' → relation_type: RECOMMENDED, negated: true\n"
    "    'Bob said we did not discuss Project Beta' → relation_type: DISCUSSED, negated: true\n"
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
    "--- EXAMPLE 4 (personal facts, subtypes) ---\n"
    'Input: "Maya Chen is a senior engineer at Atlas Corp. Her daughter Riley\n'
    '        loves jazz. Maya\'s favorite color is blue."\n'
    "Reasoning: 'senior engineer at Atlas' → HAS_TITLE(Maya→Atlas,\n"
    "subtype=senior_engineer). 'her daughter Riley' → FAMILY_OF(Maya→Riley,\n"
    "subtype=daughter). 'loves jazz' → HAS_PREFERENCE(Riley→Jazz,\n"
    "subtype=loves) or LIKES depending on vocabulary. 'favorite color is blue' →\n"
    "HAS_PREFERENCE(Maya→Blue, subtype=favorite_color).\n"
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Maya Chen", "type": "PERSON", "confidence": 0.95, "aliases": ["Maya"]},\n'
    '    {"name": "Atlas Corp", "type": "ORGANIZATION", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Riley", "type": "PERSON", "confidence": 0.9, "aliases": []},\n'
    '    {"name": "Jazz", "type": "CONCEPT", "confidence": 0.85, "aliases": []},\n'
    '    {"name": "Blue", "type": "CONCEPT", "confidence": 0.85, "aliases": []}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Maya Chen", "object": "Atlas Corp",'
    ' "relation_type": "HAS_TITLE",'
    ' "subtype": "senior_engineer", "confidence": 0.9},\n'
    '    {"subject": "Maya Chen", "object": "Riley",'
    ' "relation_type": "FAMILY_OF",'
    ' "subtype": "daughter", "confidence": 0.9},\n'
    '    {"subject": "Riley", "object": "Jazz",'
    ' "relation_type": "HAS_PREFERENCE",'
    ' "subtype": "loves_music", "confidence": 0.8},\n'
    '    {"subject": "Maya Chen", "object": "Blue",'
    ' "relation_type": "HAS_PREFERENCE",'
    ' "subtype": "favorite_color", "confidence": 0.9}\n'
    "  ]\n"
    "}\n"
    "\n"
    "--- EXAMPLE 5 (temporal, residence) ---\n"
    'Input: "The kickoff happened on 2026-03-05. Alice lives in Brooklyn now."\n'
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Kickoff", "type": "EVENT", "confidence": 0.8, "aliases": []},\n'
    '    {"name": "2026-03-05", "type": "DATETIME", "confidence": 0.95,'
    ' "aliases": []},\n'
    '    {"name": "Alice", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Brooklyn", "type": "LOCATION", "confidence": 0.95, "aliases": []}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Kickoff", "object": "2026-03-05",'
    ' "relation_type": "HAPPENED_ON", "confidence": 0.9},\n'
    '    {"subject": "Alice", "object": "Brooklyn",'
    ' "relation_type": "LIVES_IN", "confidence": 0.9}\n'
    "  ]\n"
    "}\n"
    "\n"
    "--- EXAMPLE 6 (quantified facts) ---\n"
    'Input: "Eric spent 10 hours last month watching documentaries on Netflix. '
    'He owns three bikes."\n'
    "Output:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "Eric", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
    '    {"name": "Netflix", "type": "TECHNOLOGY", "confidence": 0.9, "aliases": []},\n'
    '    {"name": "Bike", "type": "CONCEPT", "confidence": 0.85, "aliases": ["bikes"]}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject": "Eric", "object": "Netflix", "relation_type": "DISCUSSED", '
    '"subtype": "watched_documentaries", "confidence": 0.9, '
    '"quantity_value": 10, "quantity_unit": "hour", '
    '"quantity_kind": "duration", "time_scope": "last_month"},\n'
    '    {"subject": "Eric", "object": "Bike", "relation_type": "HAS_ATTRIBUTE", '
    '"subtype": "owned_count", "confidence": 0.9, '
    '"quantity_value": 3, "quantity_unit": "bike", "quantity_kind": "count"}\n'
    "  ]\n"
    "}\n"
    "\n"
    "Now extract from the following text:"
)


def _should_disable_thinking() -> bool:
    profile = LLM_PROFILES.get(settings.llm_profile)
    return profile is not None and not profile.thinking


def _thinking_enabled() -> bool | None:
    profile = LLM_PROFILES.get(settings.llm_profile)
    return profile.thinking if profile is not None else None


def _num_ctx() -> int:
    profile = LLM_PROFILES.get(settings.llm_profile)
    return profile.num_ctx if profile is not None else 8192


@traced(name="extraction.extract")
def extract(text: str) -> Extraction:
    client = ollama.Client(host=settings.ollama_url)
    prompt = f"{_SYSTEM_PROMPT}\n\n{text}"
    if _should_disable_thinking():
        prompt = "/no_think\n" + prompt
    response = client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        format=Extraction.model_json_schema(),
        think=_thinking_enabled(),
        options={"num_ctx": _num_ctx()},
    )
    prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
    completion_tokens = getattr(response, "eval_count", 0) or 0
    increment_ollama_tokens(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    record_token_usage(
        settings.llm_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return Extraction.model_validate_json(response.message.content)
