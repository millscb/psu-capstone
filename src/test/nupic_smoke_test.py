try:
    import nupic
    print("NuPIC import OK, version:", getattr(nupic, "__version__", "unknown"))
except Exception as e:
    print("NuPIC import failed:", repr(e))
    raise
