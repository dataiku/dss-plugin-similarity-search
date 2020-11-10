# -*- coding: utf-8 -*-
import pandas as pd


def reshape_nns(nns, query_names, index_names):
    """
    It expands the list of nss indices into separate rows.
    It looks up the indices for both the query series as well as the index series.
    """
    df = pd.DataFrame(data=nns[:, None].tolist(), columns=["nns"], index=query_names)
    df = df["nns"].apply(pd.Series).stack()
    df = df.reset_index().drop(columns=["level_1"])
    df = df.rename(columns={query_names.name: "query_id", 0: "nns"})
    df["nns_id"] = index_names[df.nns].values
    return df[["query_id", "nns_id"]]
