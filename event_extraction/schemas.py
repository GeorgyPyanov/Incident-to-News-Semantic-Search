from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IncidentData:
    """Structured facts extracted from an incident log."""

    original_log: str = ""
    entities: tuple[str, ...] = field(default_factory=tuple)
    locations: tuple[str, ...] = field(default_factory=tuple)
    event_types: tuple[str, ...] = field(default_factory=tuple)
    dates: tuple[str, ...] = field(default_factory=tuple)
    products: tuple[str, ...] = field(default_factory=tuple)
    services: tuple[str, ...] = field(default_factory=tuple)
    error_descriptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_structured_facts(self) -> bool:
        return any(
            (
                self.entities,
                self.locations,
                self.event_types,
                self.dates,
                self.products,
                self.services,
                self.error_descriptions,
            )
        )
