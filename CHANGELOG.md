# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-26

Bug fixes from a code review, each verified against the source and covered by a
regression test (failing on the old code, passing on the fix). Also adds the
InfluxDB 3 Core backend, GitHub Actions CI, and fixes the packaging so a
non-editable `pip install` works. Requires `amulog>=0.4.0`.

### Changed
- **`pdtimestamp*` tz helpers (internal refactor)**: the timezone-conversion
  helpers were defined identically in `source.convert` and as static methods on
  `sqlts.TimeSeriesDB`; the static methods now delegate to `source.convert` so
  the tz convention has a single source of truth (subclasses, incl. influx,
  keep inheriting them)
- **pcalg skeleton args (internal refactor)**: the `pcalg.estimate_skeleton`
  kwargs were built identically in three places (`pc_input.estimate_skeleton` /
  `estimate_dag` and `mixedlingam_input.estimate`); extracted into
  `pc_input._build_skeleton_args` (no behavior change)
- **Keep the pre-filter (raw) series by default**: new `[general] dump_org`
  option (default true) stores `log_org` (raw) alongside `log_feature`
  (filtered) in the same evdb, as separate measures/tables; the `--org` CLI flag
  still forces it on for a single run
- **evgen_log write**: commit the evdb once per read window instead of once per
  series (was ~2N fsyncs with org + feature) — preprocessing I/O speedup

### Fixed

#### critical
- **Leftover debuggers**: removed `pdb.set_trace()` left in the main pipeline
  (`log2event.load_event_log_all`) and in `source.evgen_log.LogEventLoader.details`
- **pknowledge prune-unconnected**: `_update_edge_prune_unconnected` called a
  non-existent `nx.Graph.has_path` method (now `nx.has_path(G, s, d)`) and was
  missing `return pk`, so `update()` overwrote `pk` with `None`
- **showdag.evdef2node**: returned `graph.get_node_data(node)`, which networkx
  has no such method; now `graph.nodes[node]`. Reachable via `pknowledge` and
  `visual.comparison`
- **filter_log `_resize_input`**: the shrink branch returned a list of booleans
  instead of filtering datetimes (companion to the 737928c fix of the grow branch)
- **filter_log `remove_linear`**: gated on the resized input but computed the
  cumulative curve / normalization from the pre-resize data; now consistent with
  the other filters (`discretize_sequential` + `np.cumsum`)
- **edge_search.dag_anomaly_score**: summed the `(edge, value)` generator directly
  (`int + tuple` TypeError) and shadowed the `score` argument inside the loop
- **edge_search.get_evpair_count**: applied `len()` to an int Counter value
- **edge_search.edges_anomaly_score**: `feature="edge"`/`score="idf"` called
  `get_tfidf` instead of `get_idf` (copy-paste)
- **edge_search.DAGSimilarity.similarity**: passed 1-D Series to
  `cosine_similarity` (needs 2-D) and returned a matrix instead of a scalar
- **comparison.edge_direction_diff**: removed a dead loop, fixed the
  `evdef2node` tuple unpacking that made the direction check always false, and an
  `UnboundLocalError` when `args_in_time` was empty

#### major
- **`raise Warning(...)` anti-pattern**: `cdt_input.estimate` and
  `lingam_input.estimate` raised the `Warning` class (halting) where a non-fatal
  `warnings.warn` was intended
- **EventDefinitionMap.load bare `except`**: the old-path compatibility fallback
  caught everything (incl. KeyboardInterrupt / SystemExit); narrowed to
  `except Exception:` so an interrupt during load propagates instead of silently
  falling through to the legacy path
- **make-evdb-snmp KeyboardInterrupt handling**: the snmp store handlers caught
  Ctrl-C with a bare `pass`, so an interrupted run looked like a clean success;
  they now log a warning (cleanup via `finally: el.terminate()` is unchanged)
- **pknowledge `_update_edge_prune_force`**: had no prune logic (identical to
  `_update_edge_force`); now does both prune and force. `allow_reverse` is passed
  by keyword to `has_edge` (its 3rd positional arg is `original`, so it was
  mis-bound)
- **evgen_snmp.store_all_source**: an empty `(tags, df)` pair did `return`,
  aborting the whole method and skipping every remaining source; now `continue`
- **evpost.anomaly_if**: dropped `IsolationForest(behaviour="new")`
  (scikit-learn removed `behaviour` in 0.24)
- **eval.show_match_info**: guarded the ratio computations against
  ZeroDivisionError (no tickets / no valid tickets) via `_safe_ratio`
- **visual.draw.graph_nx**: close the pygraphviz `AGraph` in a `finally` block
- **showdag.apply_filter**: no longer mutates the caller's filter-name list
  (operated in place via `remove()`); also fixed `to_undirected` being appended
  as a bare string instead of a `(name, kwargs)` tuple
- **sqlts.get_df**: `values.nan_to_num(fill)` (tuples have no such method) is now
  `np.nan_to_num(values, nan=fill)`; the `func is None` branch returns
  time-sorted rows; `if fill:` is `if fill is not None:` so `fill=0` works
- **evgen_common.drop_features**: called `drop_measure`; the backends only
  implement `drop_measurement`
- **dtutil.discretize**: the first bin (index 0) was silently dropped — the
  guard used `sum` over an array of bin indices, so `sum([0]) == 0` skipped it;
  now `len`
- **dtutil.range_dt**: built times via `fromtimestamp(ut).replace(tzinfo=...)`,
  which shifts any non-local-tz input by the local UTC offset; now
  `fromtimestamp(ut, tz=...)`
- **log2event.merge_sync_event**: renamed a column on the caller's input
  DataFrame in place; now copies first
- **evgen_log.load_items**: crashed with `for dt in None` when the filters
  removed the whole event series; now returns early
- **showdag.number_of_edges / edges**: with an explicit `graph` they returned
  `remove_edge_duplication` (a generator) instead of a count / list, breaking
  the across-host stats
- **showdag_filter `directed` / `undirected`**: the directed graph dropped edge
  attributes (e.g. `weight`), breaking a downstream `ate_prune`
- **showdag_filter._sep_across_host**: compared `src_evdef.host ==
  dst_evdef.host`; a MultipleEventDefinition has no single `.host`, so `.host`
  raised AttributeError which a broad `except AttributeError` swallowed, making
  the across-host / within-host filter return `(None, None)` silently. Now
  compares `all_attr("host")` (defined for both single and multiple) and drops
  the masking except
- **arguments.jobname2args**: split the jobname on the first `_`, breaking the
  round-trip when the area name contained `_` (e.g. `host_xxx`); now matches the
  datetime suffix
- **arguments.dag_path / evdef_path**: returned `None` when `mkdir` raised
  (missing parent / path is a file), so callers hit `open(None)`; now always
  return the path
- **evpost diff helpers** (`root_square_diff` / `diff_abs` / `anomaly_lof` /
  `anomaly_if`): set the first diff element via `ret.iloc[0]`; `ret[0]` is a
  label assignment on a DatetimeIndex (deprecated now, adds a spurious label-0
  entry on future pandas)
- **evgen_snmp._search_feature_source**: replaced `assert len(ret) == 1`
  (stripped under `-O`, an `IndexError` on zero matches, and a misleading
  "duplicated" message) with explicit not-found / duplicated `ValueError`s
- **showdag.apply_filter**: validated the filter name with `assert` (stripped
  under `-O`) and then dispatched via `eval("showdag_filter." + name)`; now an
  explicit `ValueError` on an unknown name plus `getattr` instead of `eval`
- **__main__._parse_opt_range**: the `--range` length `assert` is now an
  explicit `ValueError` (argparse `nargs=2` already covers the CLI; this guards
  non-CLI callers under `-O`)
- **eval.add_lids_stdin**: parsed each character of a single input line
  (`[int(v) for v in input()]`); now reads whitespace-separated ids from all of
  stdin
- **makedag.make_input**: crashed with `None.dump` when `log2event.makeinput`
  returned `(None, None)` (no data loaded); now returns early
- **__main__._parse_condition**: a condition with an unknown key (e.g. a typo
  like `hots=`) was silently dropped, producing a wrong filter; now raises
  `SyntaxError`
- **showdag ate_prune duplication**: removed the dead, signed-comparison
  `LogDAG.ate_prune` method (it would drop strong negative-effect edges); the
  live `showdag_filter.ate_prune` (magnitude / `abs`) is now the single
  implementation
- **influx `_get` / `get_count`**: escape tag values in the InfluxQL string
  (a value with `'` produced a broken/injectable query); use nanosecond time
  bounds instead of truncating to whole seconds (`int(ut)` + `"s"`); `get_count`
  reads the aliased field name rather than a hard-coded `"val"`
- **TimeSeriesDB backend consistency**: the backends disagreed on empty-range
  results; unified the contract — `get_count` returns `0` (was `None` in influx)
  and `get_df(func=None)` returns `None` (was an empty DataFrame in sqlts)
- **Packaging**: the wheel/sdist shipped only the top-level `logdag` package —
  the `source` / `visual` / `eval` subpackages (`visual` and `eval` had no
  `__init__.py`) and the `data/` files (incl. the default config) were omitted,
  so a non-editable `pip install` was broken. Now uses `find_packages()` and
  ships `logdag/data/*`

### Removed
- **Dead `source/evdb.py`** (a broken "OLD FILE", unused) and the empty
  `source/_common.py` stub
- **Legacy `tsdb.py`** (696-line `TimeSeriesDB`, superseded by the `source`
  backends `sqlts` / `influx`): it was reachable only through the unregistered
  `reload-area` CLI handler, so it and the dead `reload_area` handler were
  removed. This also retires the `area` NameError, the `-O`-stripped asserts,
  and the worker-connection leak the review flagged inside it
- **Dead `dtutil` functions** (~357 lines, no live callers): `is_sep` /
  `adj_sep` / `radj_sep` (retires the `adj_sep` "duration must be < 1 day"
  footgun), `separate_periodic` / `separate_periodic_dup`, `convert_binsize`,
  and the `rand_uniform` / `rand_exp` / `rand_next_exp` generators
- **Travis CI** configuration (`.travis.yml`)

### Added
- **Regression test suite** under `tests/` for the fixes above, using stubbed
  collaborators to avoid heavy DB / DAG fixtures; plus a static guard
  (`test_source_hygiene.py`) against committing live `pdb.set_trace()` /
  `breakpoint()`
- **GitHub Actions CI**: a test workflow (pytest on Python 3.8–3.12, push / PR)
  and a tag-triggered publish workflow (PyPI trusted publishing + GitHub Release)
- **Optional InfluxDB integration tests** (`tests/integration/`,
  `docker-compose.yml`) exercising `source.influx` against a real InfluxDB 1.8;
  run in CI via a service container and skipped locally when no server/client is
  present. `docker-compose.yml` also provides an InfluxDB 3 Core container
  (port 8181, ephemeral memory store, dev-only `--without-auth`) for developing
  the future v3 backend
- **TimeSeriesDB contract (conformance) tests** (`tests/contract/`): one
  behavioural suite parametrized over the storage backends (sqlts always;
  influx_v1 when a server is reachable) — a shared safety net as backends are
  added (the v1 -> v3 migration)
- **Optional `influx` extra**: `pip install -e .[influx]` installs the InfluxDB
  v1 client needed by the `source.influx` backend and its tests
- **InfluxDB 3 Core backend** (`source.influx3`, selected with
  `general.evdb = influx3`): a TimeSeriesDB backend over the v3 HTTP API
  (SQL / Line Protocol) using stdlib `urllib` — no extra client package.
  Configured via a new `[database_influx3]` config section, wired into the
  `EventLoader` backend factory, and covered by the `influx_v3` arm of the
  contract suite. `get_df(func="sum")` densifies via the shared
  `dtutil.discretize_sequential` (identical to the sqlts backend — verified by
  a contract parity test), so v3 works in the default `ci_bin_method=sequential`
  pipeline, not just sparse reads
- **InfluxDB v2 compatibility (no new code)**: the v1 backend
  (`general.evdb = influx`) works against an InfluxDB v2 server through its v1
  compatibility API — create a DBRP mapping and a v1 auth on the v2 side (now
  documented in the `[database_influx]` config comment). Verified (add /
  get_count / get_df incl. `func="sum"`) against InfluxDB 2.7, so no dedicated
  v2 backend is needed; logdag covers v1 / v2 / v3 with two implementations
- **Contract-suite required mode**: `INFLUXDB_V1_REQUIRED` /
  `INFLUXDB_V3_REQUIRED` (or `INFLUXDB_REQUIRED`) turn an unreachable influx
  backend from a skip into a hard failure, so an intended influx run that is
  actually blocked (container down, or a sandbox cutting off localhost) fails
  loudly instead of masquerading as a pass
