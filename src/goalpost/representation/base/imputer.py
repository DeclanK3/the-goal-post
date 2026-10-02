from abc import ABC, abstractmethod

from .builder import GameTable


class ImputedDataset:
    pass


class History:
    def __init__(self):
        self.entries = []

    def add(self, date, matrix):
        self.entries.append((date, matrix))


class ImputationBuilder(ABC):
    @abstractmethod
    def build(self, tables: GameTable, history: History) -> ImputedDataset:
        pass
