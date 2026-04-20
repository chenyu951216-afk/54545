from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class NotificationMessage:
    subject: str
    content: str
    metadata: dict


@dataclass(slots=True)
class NotificationResult:
    status: str
    detail: str | None = None
    attempts: int = 0


class BaseNotifier(ABC):
    channel_name: str

    @abstractmethod
    def send(self, *args, **kwargs) -> NotificationResult:
        raise NotImplementedError
