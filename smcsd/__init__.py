__all__ = ["SMCEngine"]


def __getattr__(name):
    # Lazy import: pure-torch submodules (e.g. smcsd.core.exact_accept) stay
    # importable without the sglang/zmq stack that SMCEngine pulls in.
    if name == "SMCEngine":
        from smcsd.engine import SMCEngine

        return SMCEngine
    raise AttributeError(f"module 'smcsd' has no attribute {name!r}")
