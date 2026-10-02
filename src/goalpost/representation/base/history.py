from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime

import numpy as np


@dataclass
class History:
    entries: List[Tuple[datetime, np.ndarray]] = field(default_factory=list)

    def add(self, date: datetime, matrix: np.ndarray):
        self.entries.append((date, matrix))
        self.entries.sort(key=lambda x: x[0])
