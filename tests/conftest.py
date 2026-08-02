from mojo_mapbox_earcut._lib import build


def pytest_sessionstart(session):
    build()
