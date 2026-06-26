import datetime

from .variable import EVENT_TYPES

DEFAULT_DEFAULT_TYPE = "variable"


def default_setup(g, defaults):
    if "default_type" not in defaults:
        defaults["default_type"] = DEFAULT_DEFAULT_TYPE

    s_type = set([])
    for _, node_data in g.nodes(data=True):
        if "type" in node_data:
            s_type.add(node_data["type"])
        else:
            s_type.add(defaults["default_type"])

    has_event_type = any(t in s_type for t in EVENT_TYPES)
    if "variable_index" not in defaults:
        if has_event_type:
            if "dt_range" in defaults:
                dt_range = defaults["dt_range"]
            else:
                dt_range = (datetime.datetime(2112, 9, 3),
                            datetime.datetime(2112, 9, 4))
                defaults["dt_range"] = dt_range
            if "dt_interval" in defaults:
                dt_interval = defaults["dt_interval"]
            else:
                dt_interval = datetime.timedelta(minutes=1)
                defaults["dt_interval"] = dt_interval
            if "delay" not in defaults:
                defaults["delay"] = datetime.timedelta(0)
            n_bins = int((dt_range[1] - dt_range[0]).total_seconds() / dt_interval.total_seconds())
            index = [dt_range[0] + dt_interval * diff for diff in range(n_bins)]
        else:
            index = list(range(1000))
        defaults["variable_index"] = index

    if "variable" in s_type:
        defaults.update({"variable_dist": "gaussian"})
    if "binary" in s_type:
        defaults.update({"binary_prob": 0.5})
    if "countable" in s_type or has_event_type:
        defaults.update({"poisson_lambd": 0.1})

    return defaults
