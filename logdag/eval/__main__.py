#!/usr/bin/env python
# coding: utf-8

import logging
from collections import defaultdict

from amulog import cli
from amulog import common
from amulog import config
from logdag import arguments

_logger = logging.getLogger(__package__)


def open_logdag_config(ns):
    from logdag import arguments
    return arguments.open_logdag_config(ns.conf_path, debug=ns.debug)


def add_trouble(ns):
    conf = open_logdag_config(ns)

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    tr = tm.add(ns.date, ns.group, ns.title)
    print(tr.tid)


def add_lids(ns):
    conf = open_logdag_config(ns)

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    tm.add_lids(ns.tid, ns.lids)


def add_lids_stdin(ns):
    conf = open_logdag_config(ns)

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    lids = [int(v) for v in input()]
    tm.add_lids(ns.tid, lids)


def label_trouble(ns):
    conf = open_logdag_config(ns)

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    tm.update(ns.tid, group=ns.group)


def list_trouble(ns):
    conf = open_logdag_config(ns)

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    for tr in tm:
        print(tr)


def list_group(ns):
    conf = open_logdag_config(ns)
    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    d = defaultdict(list)
    for tr in tm:
        group = tr.data["group"]
        d[group].append(tr)

    table = [["group", "ticket", "messages"]]
    num_sum = 0
    for group, l_tr in d.items():
        num = sum([len(tr.data["message"]) for tr in l_tr])
        # l_buf.append("{0}: {1} tickets ({2} messages)".format(
        #    group, len(l_tr), num))
        table.append([group, len(l_tr), num])
        num_sum += num
    table.append(["total", sum([len(v) for v in d.values()]), num_sum])
    print(common.cli_table(table))


def show_lids(ns):
    conf = open_logdag_config(ns)
    tid = ns.tid

    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    tr = tm[tid]
    print(tr)
    print("\n".join(tr.get()))


def show_trouble(ns):
    conf = open_logdag_config(ns)
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    tid = ns.tid

    from . import trouble
    tm = trouble.init_trouble_manager(conf)
    from amulog import log_db
    ld = log_db.LogData(amulog_conf)

    tr = tm[tid]
    print(tr)
    print("\n".join(tr.get_message(ld, show_lid=ns.lid_header)))


def show_trouble_info(ns):
    conf = open_logdag_config(ns)
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    tid = ns.tid

    from . import trouble
    tm = trouble.init_trouble_manager(conf)
    from amulog import log_db
    ld = log_db.LogData(amulog_conf)
    gid_name = conf.get("database_amulog", "event_gid")

    tr = tm[tid]
    d_ev, d_gid, d_host = trouble.event_stat(tr, ld, gid_name)
    d_group = trouble.event_label(d_gid, ld, gid_name)

    print(tr)
    print("{0} related events".format(len(d_ev)))
    print("{0} related hosts: {1}".format(len(d_host), sorted(d_host.keys())))
    print("{0} related templates: {1}".format(len(d_gid),
                                              sorted(d_gid.keys())))
    for group, l_gid in d_group.items():
        num = sum([d_gid[gid] for gid in l_gid])
        print("  group {0}: {1} messages, {2} templates {3}".format(
            group, num, len(l_gid), l_gid))


def list_trouble_label(ns):
    conf = open_logdag_config(ns)
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    from . import trouble
    tm = trouble.init_trouble_manager(conf)
    from amulog import log_db
    ld = log_db.LogData(amulog_conf)
    gid_name = conf.get("database_amulog", "event_gid")

    for tr in tm:
        d_ev, d_gid, d_host = trouble.event_stat(tr, ld, gid_name)
        d_group = trouble.event_label(d_gid, ld, gid_name)

        buf = "{0} ({1}): ".format(tr.tid, tr.data["group"])
        for group, l_gid in sorted(d_group.items(), key=lambda x: len(x[1]),
                                   reverse=True):
            num = sum([d_gid[gid] for gid in l_gid])
            buf += "{0}({1},{2}) ".format(group, len(l_gid), num)
        print(buf)


def list_trouble_stat(ns):
    conf = open_logdag_config(ns)
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    from . import trouble
    tm = trouble.init_trouble_manager(conf)
    from amulog import log_db
    ld = log_db.LogData(amulog_conf)
    gid_name = conf.get("database_amulog", "event_gid")

    from scipy.stats import entropy

    table = [["trouble_id", "group", "messages", "gids", "hosts",
              "events", "groups",
              "entropy_events", "entropy_groups"]]
    for tr in tm:
        line = []
        d_ev, d_gid, d_host = trouble.event_stat(tr, ld, gid_name)
        d_group = trouble.event_label(d_gid, ld, gid_name)
        ent_ev = entropy(list(d_ev.values()), base=2)
        ent_group = entropy([sum([d_gid[gid] for gid in l_gid])
                             for l_gid in d_group.values()],
                            base=2)
        line.append(tr.tid)
        line.append(tr.data["group"])
        line.append(sum(d_gid.values()))  # messages
        line.append(len(d_gid.keys()))  # gids
        line.append(len(d_host.keys()))  # hosts
        line.append(len(d_ev.keys()))  # events
        line.append(len(d_group.keys()))  # groups
        line.append(ent_ev)  # entropy of events
        line.append(ent_group)  # entropy of groups
        table.append(line)

    print(common.cli_table(table))


def show_match(ns):
    conf = open_logdag_config(ns)
    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    from logdag import showdag
    from . import match_edge
    tr = tm[ns.tid]
    d_args = match_edge.match_edges(conf, tr, rule=ns.rule, cond=ns.cond)
    cnt = sum([len(l_edge) for l_edge in d_args.values()])
    print("{0[date]} ({0[group]}): {1}".format(tr.data, cnt))
    for name, l_edge in d_args.items():
        r = showdag.LogDAG(arguments.name2args(name, conf))
        r.load()
        for edge in l_edge:
            edgestr = r.edge_str(edge, graph=r.graph.to_undirected())
            print(name, edgestr)


def show_match_all(ns):
    conf = open_logdag_config(ns)
    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    from . import match_edge
    for tr in tm:
        d_args = match_edge.match_edges(conf, tr, rule=ns.rule,
                                        cond=ns.cond)
        cnt = sum([len(l_edge) for l_edge in d_args.values()])
        print("Trouble {0.tid} {0.data[date]} ({0.data[group]}): {1}".format(
            tr, cnt))


def show_match_diff(ns):
    l_conffp = ns.confs
    assert len(l_conffp) == 2
    kwargs = {"ex_defaults": [arguments.DEFAULT_CONFIG]}
    conf1 = config.open_config(l_conffp[0], **kwargs)
    conf2 = config.open_config(l_conffp[1], **kwargs)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from . import trouble
    tm = trouble.init_trouble_manager(conf1)

    def _dag_from_name(conf, name):
        args = arguments.name2args(name, conf)
        r = showdag.LogDAG(args)
        r.load()
        return r

    from logdag import showdag
    from . import match_edge
    for tr in tm:
        d_args1 = match_edge.match_edges(conf1, tr, rule=ns.rule,
                                         cond=ns.cond)
        cnt1 = sum([len(l_edge) for l_edge in d_args1.values()])
        d_args2 = match_edge.match_edges(conf2, tr, rule=ns.rule,
                                         cond=ns.cond)
        cnt2 = sum([len(l_edge) for l_edge in d_args2.values()])
        if cnt1 == cnt2:
            pass
        else:
            print("Trouble {0} {1} ({2})".format(
                tr.tid, tr.data["date"], tr.data["group"]))
            print("{0}: {1}".format(config.getname(conf1), cnt1))
            for key, l_edge in d_args1.items():
                r1 = _dag_from_name(conf1, key)
                for edge in l_edge:
                    edgestr = r1.edge_str(edge, graph=r1.graph.to_undirected())
                    print(key, edgestr)
            print("{0}: {1}".format(config.getname(conf2), cnt2))
            for key, l_edge in d_args2.items():
                r2 = _dag_from_name(conf2, key)
                for edge in l_edge:
                    edgestr = r2.edge_str(edge, graph=r2.graph.to_undirected())
                    print(key, edgestr)
            print("")


def show_match_info(ns):
    conf = open_logdag_config(ns)
    from . import trouble
    tm = trouble.init_trouble_manager(conf)

    from . import match_edge
    valid_cnt = 0
    d_num = {}
    for tr in tm:
        if tr.data["group"] != trouble.EMPTY_GROUP:
            try:
                d_args = match_edge.match_edges(conf, tr, rule=ns.rule,
                                                cond=ns.cond)
                cnt = sum([len(l_edge) for l_edge in d_args.values()])
                d_num[tr.tid] = cnt
                valid_cnt += 1
            except IOError:
                msg = "DAG incompleted for ticket {0}, passed".format(tr.tid)
                _logger.info(msg)

    match_edge_sum = sum(d_num.values())
    # valid_ticket_num = sum([1 for tr in tm
    #                         if not tr.data["group"] == trouble.EMPTY_GROUP])
    detected_ticket_num = sum([1 for v in d_num.values() if v > 0])

    valid_ratio = valid_cnt / len(tm)
    print("valid: {0} in {1} ({2})".format(valid_cnt,
                                           len(tm),
                                           valid_ratio))
    detected_ratio = detected_ticket_num / valid_cnt
    print("detected: {0} in {1} ({2})".format(detected_ticket_num,
                                              valid_cnt,
                                              detected_ratio))
    print("average edges: {0}".format(1.0 * match_edge_sum / valid_cnt))


def search_trouble(ns):
    conf = open_logdag_config(ns)
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    d = parse_condition(ns.conditions)
    from . import trouble
    tm = trouble.init_trouble_manager(conf)
    from amulog import log_db
    ld = log_db.LogData(amulog_conf)
    gid_name = conf.get("database_amulog", "event_gid")

    # match group
    if "group" in d:
        l_tr = [tr for tr in tm if tr.data["group"] == d["group"]]
    else:
        l_tr = [tr for tr in tm]

    # match event
    if "gid" in d or "host" in d:
        search_gid = d.get("gid", None)
        search_host = d.get("host", None)
        ret = []
        for tr in l_tr:
            for lid in tr.data["message"]:
                lm = ld.get_line(lid)
                gid = lm.lt.get(gid_name)
                host = lm.host
                if (search_gid is None or search_gid == gid) and \
                        (search_host is None or search_host == host):
                    ret.append(tr)
                    break
        l_tr = ret

    for tr in l_tr:
        print(tr)


def parse_condition(conditions):
    """
    Args:
        conditions (list)
    """
    d = {}
    for arg in conditions:
        if "=" not in arg:
            raise SyntaxError
        key = arg.partition("=")[0]
        if key == "gid":
            d["gid"] = int(arg.partition("=")[-1])
        elif key == "host":
            d["host"] = arg.partition("=")[-1]
        elif key == "group":
            d["group"] = arg.partition("=")[-1]
        else:
            d[key] = arg.partition("=")[-1]
    return d


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_LID = [["-l", "--lid"],
           {"dest": "lid_header", "action": "store_true",
            "help": "parse lid from head part of log message"}]
OPT_RULE = [["-r", "--rule"],
            {"dest": "rule", "action": "store",
             "type": str, "default": "all",
             "help": "one of [all, either, both, log-snmp]"}]
OPT_CONDITION = [["-e", "--edge-condition"],
                 {"dest": "cond", "action": "store",
                  "type": str, "default": None,
                  "help": "rules for matched edges, [xhost] is now available"}]
ARG_TID = [["tid"],
           {"metavar": "TID", "action": "store", "type": int,
            "help": "trouble identifier"}]
ARG_DATE = [["date"],
            {"metavar": "DATE", "action": "store",
             "help": "%%Y%%m%%d style date string"}]
ARG_GROUP = [["group"],
             {"metavar": "GROUP", "action": "store",
              "help": "class of the trouble"}]
ARG_TITLE = [["title"],
             {"metavar": "TITLE", "action": "store",
              "help": "recorded title of the trouble"}]
ARG_MESSAGES = [["lids"],
                {"metavar": "MESSAGE_IDS", "action": "store",
                 "type": int, "nargs": "+",
                 "help": "message IDs (lid in amulog db)"}]
ARG_SEARCH = [["conditions"],
              {"metavar": "CONDITION", "nargs": "+",
               "help": ("Conditions to search log messages. "
                        "Example: command gid=24 group=system ..., "
                        "Keys: gid, host, group")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "add-trouble": ["Add a new trouble definition",
                    [OPT_CONFIG, OPT_DEBUG, ARG_DATE, ARG_GROUP, ARG_TITLE],
                    add_trouble],
    "add-lids": ["Add messages to a trouble",
                 [OPT_CONFIG, OPT_DEBUG, ARG_TID, ARG_MESSAGES],
                 add_lids],
    "add-lids-stdin": ["Add messages to a trouble from stdin",
                       [OPT_CONFIG, OPT_DEBUG, ARG_TID],
                       add_lids_stdin],
    "label-trouble": ["Set label to a trouble",
                      [OPT_CONFIG, OPT_DEBUG, ARG_TID, ARG_GROUP],
                      label_trouble],
    "list-trouble": ["List all troubles",
                     [OPT_CONFIG, OPT_DEBUG],
                     list_trouble],
    "list-group": ["List group information",
                   [OPT_CONFIG, OPT_DEBUG],
                   list_group],
    "list-trouble-stat": ["List stats of messages in each troubles",
                          [OPT_CONFIG, OPT_DEBUG],
                          list_trouble_stat],
    "list-trouble-label": ["Show corresponding label groups of tickets",
                           [OPT_CONFIG, OPT_DEBUG],
                           list_trouble_label],
    "show-lids": ["Show all lids in the given trouble",
                  [OPT_CONFIG, OPT_DEBUG, ARG_TID],
                  show_lids],
    "show-trouble": ["Show all messages corresponding to the given trouble",
                     [OPT_CONFIG, OPT_DEBUG, OPT_LID, ARG_TID],
                     show_trouble],
    "show-trouble-info": ["Show abstracted information for the trouble",
                          [OPT_CONFIG, OPT_DEBUG, ARG_TID],
                          show_trouble_info],
    "search-trouble": ["Search troubles with messages of specified features",
                       [OPT_CONFIG, OPT_DEBUG, ARG_SEARCH],
                       search_trouble],
    "show-match": ["Show matching edges with a ticket",
                   [OPT_CONFIG, OPT_DEBUG, OPT_RULE, OPT_CONDITION, ARG_TID],
                   show_match],
    "show-match-all": ["Show matching edges with all tickets",
                       [OPT_CONFIG, OPT_DEBUG, OPT_RULE, OPT_CONDITION],
                       show_match_all],
    "show-match-diff": ["Compare 2 configs with all tickets",
                        [OPT_DEBUG, OPT_RULE, OPT_CONDITION,
                         [["confs"],
                          {"metavar": "CONFIG", "nargs": 2,
                           "help": "2 config file path"}], ],
                        show_match_diff],
    "show-match-info": ["Show abstracted information of edges in all DAG",
                        [OPT_CONFIG, OPT_DEBUG, OPT_RULE, OPT_CONDITION],
                        show_match_info]
}


def main():
    cli.main(DICT_ARGSET)


if __name__ == "__main__":
    main()
