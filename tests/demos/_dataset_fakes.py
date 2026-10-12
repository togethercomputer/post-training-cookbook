"""A stand-in for the `datasets` splits the demos load."""

from __future__ import annotations


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return Rows([self.rows[i] for i in indices])

    def __getitem__(self, key):
        return [row[key] for row in self.rows]
