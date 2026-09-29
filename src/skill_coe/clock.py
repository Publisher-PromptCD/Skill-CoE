"""Real elapsed time, independent of the benchmark's simulated calendar."""
import time


class _NativeClock:
    # Store callables inside a class: freezegun rewrites module attributes.
    monotonic = staticmethod(time.monotonic)
    clock_gettime = staticmethod(getattr(time, "clock_gettime", None))
    clock_id = getattr(time, "CLOCK_MONOTONIC", None)


def elapsed_clock():
    if _NativeClock.clock_gettime is not None and _NativeClock.clock_id is not None:
        return _NativeClock.clock_gettime(_NativeClock.clock_id)
    return _NativeClock.monotonic()
