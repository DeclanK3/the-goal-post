from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from .models import Game


class DataSource(ABC):
    @abstractmethod
    def fetch(self, seasons: List[int]) -> None:
        pass

    @abstractmethod
    def fetch_upcoming(self, as_of: datetime) -> List[Game]:
        pass

    @abstractmethod
    def parse(self) -> List[Game]:
        pass

    def load(self, seasons: List[int]) -> List[Game]:
        self.fetch(seasons)
        return self.parse()
