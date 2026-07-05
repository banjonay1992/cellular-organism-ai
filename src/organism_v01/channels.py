from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChannelLayout:
    """Named channel layout for the organism state tensor.

    Environment channels are clamped by the task. Mutable channels are updated
    by the shared cell network.
    """

    hidden_channels: int = 16
    route_channels: int = 0

    source_a: int = 0
    source_b: int = 1
    sink: int = 2
    blocked: int = 3
    alive: int = 4
    x_field: int = 5
    y_field: int = 6

    @property
    def env_count(self) -> int:
        return 7 + self.route_channels

    @property
    def route_start(self) -> int:
        return 7

    @property
    def route_slice(self) -> slice:
        return slice(self.route_start, self.route_start + self.route_channels)

    @property
    def hidden_start(self) -> int:
        return self.env_count

    @property
    def output_start(self) -> int:
        return self.hidden_start + self.hidden_channels

    @property
    def output_a(self) -> int:
        return self.output_start

    @property
    def output_b(self) -> int:
        return self.output_start + 1

    @property
    def output_count(self) -> int:
        return 2

    @property
    def total_channels(self) -> int:
        return self.output_start + self.output_count

    @property
    def hidden_slice(self) -> slice:
        return slice(self.hidden_start, self.output_start)

    @property
    def output_slice(self) -> slice:
        return slice(self.output_start, self.total_channels)

    @property
    def mutable_slice(self) -> slice:
        return slice(self.hidden_start, self.total_channels)

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data["env_count"] = self.env_count
        data["total_channels"] = self.total_channels
        data["output_start"] = self.output_start
        return data
