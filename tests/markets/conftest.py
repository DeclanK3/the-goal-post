"""Market bridge tests never use live odds, public feeds or message routes."""

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Market bridge tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
