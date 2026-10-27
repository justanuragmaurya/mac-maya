"""Maya TTS inference for Apple Silicon (MPS) and CPU, built on the maya-research/maya2-native weights."""
__all__ = ["MayaTTS", "EngineConfig"]


def __getattr__(name):
    if name in __all__:
        from maya_mac import engine
        return getattr(engine, name)
    raise AttributeError(name)


def main() -> None:
    from maya_mac.cli import main as cli_main
    cli_main()
