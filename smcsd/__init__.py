__all__ = ["SMCEngine"]


def __getattr__(name):
    if name == "SMCEngine":
        from smcsd.engine import SMCEngine

        return SMCEngine
    raise AttributeError(name)
