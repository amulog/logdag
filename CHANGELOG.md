# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-08-05

Bugfix release. Restores installability on Python 3.13/3.14, which lingam's
dependency pin had broken.

### Added
- **`tutorial/ctd_tsevent.ipynb`**: the causaltestdata tutorial notebook, carried
  over from the standalone `causaltestdata` repository (which is to be archived)
  when its code was vendored as `logdag.causaltestdata` in 0.3.0. It generates
  time-series events from a known DAG and recovers the structure with PC and
  LiNGAM; imports now target the vendored subpackage.

### Changed
- **`lingam` moved from the core requirements to an optional extra**: install it
  with `pip install logdag[lingam]` to use `cause_algorithm = lingam` or
  `lingam-corr`. lingam depends on semopy, whose `polycorr` module calls
  `scipy.stats.mvn.mvnun` -- an attribute the scipy shim stopped exposing in
  1.14 -- so lingam pins `scipy<=1.13.1`. That scipy predates Python 3.13 and
  ships no wheel for it, so installing logdag on 3.13/3.14 fell back to building
  scipy from source and failed for want of OpenBLAS. logdag itself only uses
  ICALiNGAM, DirectLiNGAM and `make_prior_knowledge`, none of which reach the
  semopy path, so the pin buys us nothing and the dependency is better declared
  where users can opt into it. `lingam_input` now raises an `ImportError` that
  names the extra when the package is missing.
  The extra carries a `python_version < "3.13"` marker (as `testdata` already
  did), so requesting it on 3.13/3.14 installs nothing instead of failing a scipy
  source build.
- CI installs the `lingam` and `testdata` extras in the unit-test matrix, and sets
  `LINGAM_REQUIRED=1` on the versions where `lingam` resolves, so its recovery
  test fails loudly there instead of skipping unnoticed everywhere. Previously no
  job installed either extra.

### Fixed
- **`logdag` subcommand name typo**: the subcommand that prints configuration
  defaults is now `show-default-config` (was `show-deafult-config`). The old
  misspelled name is no longer accepted.

## [0.3.1] - 2026-07-02

Planned patch release. Removes the non-functional built-in `mixedlingam`
(it required the non-public `bcause` package) and adds an out-of-tree plugin
seam so such methods can be provided privately.

### Added
- **Out-of-tree algorithm plugins**: `makedag.estimate_dag` resolves an unknown
  `[dag] cause_algorithm` through the `logdag.cause_algorithm` entry-point group,
  calling a plugin's `estimate(conf, input_df, prior_knowledge=None) -> DiGraph`.
  This lets methods that cannot ship publicly (e.g. wrappers around a private
  library) be installed and selected without their code or name living in the
  public tree.

### Removed
- **Built-in `mixedlingam` support**: dropped `mixedlingam_input.py` and its
  `cause_algorithm` branch. Its MixedLiNGAM implementation depends on the
  non-public `bcause` package, so `mixedlingam` now lives out-of-tree and is
  provided through the `logdag.cause_algorithm` plugin mechanism above (kept
  private for paper reproducibility; install the plugin to use
  `cause_algorithm = mixedlingam`).

## [0.3.0] - 2026-06-29

Minor release coordinated with amulog 0.5.0 (host grouping). Vendors the
causaltestdata generator into the tree (`logdag.causaltestdata`), consumes
amulog's host stratification so events can be aggregated by host-group tier
(e.g. BGL chip `R02-M1-N0-C:...` -> midplane `R02-M1`), adds Python 3.13/3.14
support, and fixes two evdb/loader bugs found along the way. Backward
compatible: host aggregation is off by default (empty `host_tier` keeps the
original per-host behaviour) and no database rebuild is needed. Requires
`amulog>=0.5.0`.

### Added
- **Python 3.13 / 3.14 support**: the core and all its dependencies
  (numpy, scipy, pandas, scikit-learn, statsmodels, lingam, pcalg, gsq) install
  and pass on 3.13/3.14; CI runs 3.8-3.14. The optional `testdata` extra is the
  one exception -- `Hawkes` ships no 3.13+ wheel, so it is constrained to
  `python_version < "3.13"` (it is needed only by `HawkesEventVariable`, which
  the core and tests never use).
- **Vendored `logdag.causaltestdata`**: merged the standalone `causaltestdata`
  package (BSD-3-Clause, same author) into the source tree as a sub-package.
  It generates synthetic time-series from a known causal DAG, used for
  test/evaluation data with ground-truth structure. Includes a new
  `PeriodicEventVariable` (`type = "periodic"`) producing regular-interval event
  series -- the kind the default preprocessing filters (`filter_periodic` /
  `remove_linear`) remove, so a synthetic model can mix periodic (dropped) and
  causal Poisson (kept) events. Its only non-core dependency, `Hawkes` (needed
  solely by `HawkesEventVariable`), is imported lazily and declared as the
  `testdata` extra. The package is for tests / evaluation / downstream consumers
  (e.g. logdagviz) only; logdag's core does not import it.
- **`logdag.causaltestdata.amulog_export`**: bridge that turns a synthetic DAG's
  event series into amulog-style log lines
  (`"YYYY-MM-DD HH:MM:SS host message"`), so a known causal structure can drive
  the amulog -> evdb pipeline. Backed by a new `variable.generate_variables`
  (the variable-building half of `generate_all`, factored out so callers can
  reach each node's `.ts`). Adds an end-to-end test
  (`tests/test_causaltestdata_pipeline.py`) that builds an amulog DB from a
  known DAG and checks the default filters drop the periodic series while a
  sparse Poisson series survives -- the first step of consolidating logdag's
  test fixtures onto ground-truth-structure data.
- **Host stratification (amulog host_group) consumption**: new
  `[database_amulog] host_tier` option. When non-empty, `src_amulog.AmulogLoader`
  resolves each original host to a host group id at that tier (via amulog's
  `host_group` resolver), aggregating events by host group -- e.g. BGL chip
  hosts `R02-M1-N0-C:...` collapse to midplane `R02-M1`. The amulog log table is
  untouched (host stays original); aggregation is done on the fly, so no DB
  rebuild. Empty `host_tier` (the default) keeps the legacy per-host behaviour
  unchanged. Requires amulog with `host_group` (the tier must be defined in the
  amulog `[manager] host_group_filename`).
  A full E2E test (`tests/test_hostgroup_pipeline.py`) drives a known DAG over
  synthetic multi-layer (BGL-style) hosts through amulog and make-dag, checking
  that `host_tier=midplane` merges chip-host series and still recovers the
  cause->effect edge between aggregated nodes (synthetic hosts only, no loghub
  data).

### Fixed
- **`sqlts.SQLTimeSeries.drop_measurement`**: called a non-existent `drop_sql`
  on amulog's DB helper (`AttributeError`); now uses `drop_table_sql` and skips
  a measure whose table was never created, so `drop_features()` works.
- **`src_amulog.init_amulogloader`**: passed the `source_conf` path and
  `dt_range` in the wrong constructor positions (so `conf` received `dt_range`
  and the loader read by `eval/match_edge.py` was misconfigured); now opens the
  amulog config via `config.open_config` and wires arguments correctly (also
  forwarding `host_tier`).

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
  ships `logdag/data/*`. `MANIFEST.in` also ships `requirements.txt` (read by
  `setup.py` at build time — without it the wheel build from the sdist failed
  with `FileNotFoundError`) and references the actual `README.rst`

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
