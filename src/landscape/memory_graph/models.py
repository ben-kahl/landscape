from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssertionPayload:
    source_kind: str
    source_id: str
    raw_subject_text: str
    raw_relation_text: str
    raw_object_text: str
    confidence: float
    subtype: str | None = None
    family_candidate: str | None = None
    value_text: str | None = None
    value_number: float | None = None
    value_unit: str | None = None
    value_kind: str | None = None
    value_time: str | None = None
    quantity_value: float | str | None = None
    quantity_unit: str | None = None
    quantity_kind: str | None = None
    time_scope: str | None = None
    negated: bool = False
    chunk_refs: list[tuple[str, int | None, int | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep the legacy quantity fields and the new unified value fields in sync,
        # but reject contradictory dual representations.
        if self.value_text is not None and isinstance(self.quantity_value, str):
            if self.value_text != self.quantity_value:
                raise ValueError("value_text and quantity_value conflict")
        elif self.value_text is None and isinstance(self.quantity_value, str):
            object.__setattr__(self, "value_text", self.quantity_value)

        if self.value_number is not None and isinstance(self.quantity_value, (int, float)):
            if float(self.value_number) != float(self.quantity_value):
                raise ValueError("value_number and quantity_value conflict")
        elif self.value_number is None and isinstance(self.quantity_value, (int, float)):
            object.__setattr__(self, "value_number", float(self.quantity_value))

        if self.value_text is not None and self.value_number is not None:
            raise ValueError("value_text and value_number are mutually exclusive")

        if self.quantity_value is None:
            if self.value_number is not None:
                object.__setattr__(self, "quantity_value", self.value_number)
            elif self.value_text is not None:
                object.__setattr__(self, "quantity_value", self.value_text)
        elif self.value_text is not None and isinstance(self.quantity_value, str):
            object.__setattr__(self, "quantity_value", self.value_text)
        elif self.value_number is not None and isinstance(self.quantity_value, (int, float)):
            object.__setattr__(self, "quantity_value", self.value_number)

        if self.value_unit is None and self.quantity_unit is not None:
            object.__setattr__(self, "value_unit", self.quantity_unit)
        elif self.value_unit is not None and self.quantity_unit is not None:
            if self.value_unit != self.quantity_unit:
                raise ValueError("value_unit and quantity_unit conflict")
        if self.quantity_unit is None and self.value_unit is not None:
            object.__setattr__(self, "quantity_unit", self.value_unit)

        if self.value_kind is None and self.quantity_kind is not None:
            object.__setattr__(self, "value_kind", self.quantity_kind)
        elif self.value_kind is not None and self.quantity_kind is not None:
            if self.value_kind != self.quantity_kind:
                raise ValueError("value_kind and quantity_kind conflict")
        if self.quantity_kind is None and self.value_kind is not None:
            object.__setattr__(self, "quantity_kind", self.value_kind)

        if self.value_time is None and self.time_scope is not None:
            object.__setattr__(self, "value_time", self.time_scope)
        elif self.value_time is not None and self.time_scope is not None:
            if self.value_time != self.time_scope:
                raise ValueError("value_time and time_scope conflict")
        if self.time_scope is None and self.value_time is not None:
            object.__setattr__(self, "time_scope", self.value_time)
