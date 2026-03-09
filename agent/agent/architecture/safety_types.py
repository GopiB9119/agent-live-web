from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionClass(str, Enum):
    READ_ONLY = "read_only"
    SCOPED_REVERSIBLE_WRITE = "scoped_reversible_write"
    BROAD_LOCAL_WRITE = "broad_local_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    DESTRUCTIVE = "destructive"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_VERIFICATION = "allow_with_verification"
    PREVIEW_REQUIRED = "preview_required"
    CONFIRM_REQUIRED = "confirm_required"
    BLOCKED = "blocked"


@dataclass
class SafetyEvaluation:
    tool_name: str
    action_class: str
    risk_level: str
    decision: str
    reason_codes: list[str]
    requires_verification: bool = False
    preview_summary: dict[str, Any] | None = None
    confirm_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "tool": self.tool_name,
            "action_class": self.action_class,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "requires_verification": bool(self.requires_verification),
        }
        if self.preview_summary is not None:
            payload["preview_summary"] = self.preview_summary
        if self.confirm_token:
            payload["confirm_token"] = self.confirm_token
        return payload
