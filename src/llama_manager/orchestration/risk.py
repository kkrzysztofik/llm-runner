"""Risk acknowledgement management for server launches."""

import uuid


class RiskAckManager:
    """Tracks risk acknowledgements per launch attempt."""

    def __init__(self) -> None:
        self._risky_acknowledged_cache: dict[str, set[str]] = {}
        self._current_launch_attempt_id: str | None = None

    def begin_launch_attempt(self, attempt_id: str | None = None) -> str:
        """Create/select launch attempt and initialize per-attempt ack cache."""
        attempt_id = attempt_id or uuid.uuid4().hex
        self._current_launch_attempt_id = attempt_id
        self._risky_acknowledged_cache.setdefault(attempt_id, set())
        return attempt_id

    def issue_ack_token(self, attempt_id: str | None = None) -> str:
        """Issue deterministic ack token bound to a launch attempt."""
        attempt_id = attempt_id or self.begin_launch_attempt()
        return f"ack:{attempt_id}"

    def validate_ack_token(self, attempt_id: str, ack_token: str | None) -> bool:
        """Validate that ack_token is bound to attempt_id."""
        if ack_token is None:
            return False
        return ack_token == f"ack:{attempt_id}"

    def acknowledge_risk(
        self,
        slot_id: str,
        risk_type: str,
        attempt_id: str | None = None,
        ack_token: str | None = None,
    ) -> None:
        """Mark a risky operation as acknowledged for a specific slot."""
        attempt_id = attempt_id or self._current_launch_attempt_id
        if attempt_id is None:
            attempt_id = self.begin_launch_attempt()

        if ack_token is not None and not self.validate_ack_token(attempt_id, ack_token):
            raise ValueError("ack_token does not match attempt_id")

        self._risky_acknowledged_cache.setdefault(attempt_id, set()).add(f"{slot_id}:{risk_type}")

    def is_risk_acknowledged(
        self,
        slot_id: str,
        risk_type: str,
        attempt_id: str | None = None,
    ) -> bool:
        """Check if a risky operation has been acknowledged for a specific slot."""
        attempt_id = attempt_id or self._current_launch_attempt_id
        if attempt_id is None:
            return False
        return f"{slot_id}:{risk_type}" in self._risky_acknowledged_cache.get(attempt_id, set())

    def clear_all(self) -> None:
        """Clear all in-memory risk acknowledgement state."""
        self._risky_acknowledged_cache.clear()
        self._current_launch_attempt_id = None
