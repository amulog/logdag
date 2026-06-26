"""Export generated event variables as amulog-style log lines.

Bridges causaltestdata's event time-series (``Variable.ts``) to the plain-text
log format amulog parses -- ``"YYYY-MM-DD HH:MM:SS host message"``, the same
format as ``amulog.testutil.TestLogGenerator.dump_log`` -- so a synthetic DAG
with a *known* causal structure can drive the amulog -> evdb pipeline used in
logdag's tests (and by downstream consumers such as logdagviz).

The intended mapping is one DAG event-node -> one fixed log message -> one
amulog template (gid), so the generated event series of a node and the evdb
event series of its gid correspond 1:1. Non-event variables (continuous /
binary / countable, i.e. those without a ``.ts``) are skipped.
"""

import datetime

from . import variable as _variable

LOG_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_HOST = "host0"


def _resolve(spec):
    """Normalize a node_message entry to (host, message)."""
    if isinstance(spec, (tuple, list)):
        host, message = spec
        return host, message
    return DEFAULT_HOST, spec


def generate_log_events(g, defaults, node_message):
    """Generate the DAG and return its event log rows, time-sorted.

    Args:
        g: networkx.DiGraph causal DAG (passed to ``generate_variables``).
        defaults: causaltestdata defaults dict (e.g. ``dt_range``,
            ``default_type``). Mutated/augmented by ``default_setup`` as usual.
        node_message: dict node_id -> message ``str`` or ``(host, message)``.
            Only nodes present here *and* carrying a ``.ts`` produce log lines.

    Returns:
        (variables, rows) where ``variables`` is the node_id -> Variable dict
        (so callers can inspect ground truth) and ``rows`` is a list of
        ``(datetime, host, message)`` sorted by time.
    """
    _, variables = _variable.generate_variables(g, defaults)

    rows = []
    for node_id, var in variables.items():
        ts = getattr(var, "ts", None)
        if ts is None:
            # non-event variable (no event timestamps to emit)
            continue
        if node_id not in node_message:
            continue
        host, message = _resolve(node_message[node_id])
        for dt in ts:
            rows.append((dt, host, message))

    rows.sort(key=lambda x: x[0])
    return variables, rows


def format_log_line(dt, host, message):
    # amulog parses local wall-clock text; sub-second jitter is truncated by
    # the second-resolution format, matching testutil.TestLogGenerator.
    if not isinstance(dt, datetime.datetime):
        # tolerate numpy datetime64 / pandas Timestamp
        dt = datetime.datetime.fromisoformat(str(dt)[:19])
    return " ".join((dt.strftime(LOG_DT_FORMAT), host, message))


def dump_log(g, defaults, node_message, output):
    """Generate the DAG and write its event log to ``output``.

    Args:
        output: path to write, or None to print to stdout (parity with
            ``testutil.TestLogGenerator.dump_log``).

    Returns:
        (variables, rows) -- same as ``generate_log_events`` -- so a test can
        assert on the ground-truth series it just wrote.
    """
    variables, rows = generate_log_events(g, defaults, node_message)
    lines = [format_log_line(*row) for row in rows]
    if output is None:
        for line in lines:
            print(line)
    else:
        with open(output, "w") as f:
            for line in lines:
                f.write(line + "\n")
    return variables, rows
