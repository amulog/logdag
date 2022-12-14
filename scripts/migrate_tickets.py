#!/usr/bin/env python
# coding: utf-8

"""Migrate logs corresponding to trouble tickets
between two environments with same dataset but different amulog DB
(i.e., different lids or log template sets).
"""

import sys
import datetime
from tqdm import tqdm


FAILURE_OUTPUT = "migrate_failure.txt"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: {0} CONF1 CONF2".format(sys.argv[0]))

    from logdag import arguments
    conf1 = arguments.open_logdag_config(sys.argv[1])
    conf2 = arguments.open_logdag_config(sys.argv[2])

    from amulog import config
    from amulog import log_db
    amulog_conf1 = config.open_config(conf1["database_amulog"]["source_conf"])
    ld1 = log_db.LogData(amulog_conf1)
    amulog_conf2 = config.open_config(conf2["database_amulog"]["source_conf"])
    ld2 = log_db.LogData(amulog_conf2)

    from logdag.eval import trouble
    dirname1 = conf1.get("eval", "path")
    dirname2 = conf2.get("eval", "path")
    tm1 = trouble.TroubleManager(dirname1)
    tm2 = trouble.TroubleManager(dirname2)

    from amulog.lt_misc import edit_distance
    for src_tr, dst_tr in zip(tm1, tm2):
        dst_tr.data["message"] = []
        iterobj = tqdm(sorted(src_tr.data["message"]), ascii=True)
        iterobj.set_description("Ticket {0}".format(src_tr.tid))
        l_lid = []
        for lid in iterobj:
            src_lm = ld1.get_line(lid)

            kwargs = {"dts": src_lm.dt,
                      "dte": src_lm.dt + datetime.timedelta(seconds=1),
                      "host": src_lm.host}
            candidates = [(lm, edit_distance(src_lm.l_w, lm.l_w))
                          for lm in ld2.iter_lines(**kwargs)]
            if len(candidates) == 0:
                with open(FAILURE_OUTPUT, 'a') as f:
                    f.write("Ticket {0} lid {1}: {2}\n".format(
                        src_tr.tid, lid, src_lm.restore_line()
                    ))
            else:
                dst_lm = min(candidates, key=lambda x: x[1])[0]
                l_lid.append(dst_lm.lid)
        dst_tr.data["message"] += l_lid
        dst_tr.dump()
