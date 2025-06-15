# Debugging

## Configuring Logging (to see debug logs)

To see debug-level logs (e.g. for inspecting codebook updates), follow the 2 steps below:

1. Configure logging in your script by calling `configure_logging()` from `zip2zip.logging_utils`:

```python
from zip2zip.logging_utils import configure_logging


if __name__ == "__main__":
    configure_logging()

    # ... rest of your script ...
```

2. Set the environment variable `ZIP2ZIP_LOGLEVEL` before running your script:

```bash
export ZIP2ZIP_LOGLEVEL=DEBUG
python examples/debug.py
```

You can also set it inline:

```bash
ZIP2ZIP_LOGLEVEL=DEBUG python examples/debug.py
```

This controls the verbosity of internal logs. The default log level is `WARNING` if not specified.
