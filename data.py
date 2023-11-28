import clickhouse_connect

client = clickhouse_connect.get_client(
    host="localhost", username="cgoeldel", password="Goe1409ldel"
)

import networkx as nx

G = nx.DiGraph()

batch_size = 1

cs_neighbours = 5
hist_mod_neighbours = 20
gene_neighbours = 1
meth_neighbours = 20

mutables = 5  # not implemented yet

with open("nodelist.txt") as f:
    nodes = f.read().splitlines()

with open("edgelist.txt") as f:
    edges = f.read().splitlines()

for node in nodes:
    if node == "filename,node,gen,meth" or node == "":
        continue

    filename, node, gen, meth = node.split(",")

    node = {
        "filename": filename,
        "node": node,
        "gen": int(gen),
        "meth": True if meth == "Y" else False,
    }

    G.add_node(node["node"], **node)

for edge in edges:
    if edge == "from,to":
        continue

    from_, to_ = edge.split(",")

    G.add_edge(from_, to_)


def get_predecessor_node(node):
    if not node in G:
        return None

    pred = iter(G.pred[node])

    if pred.__length_hint__() == 0:
        return None

    pred = G.nodes[next(pred)]

    if pred["meth"]:
        return pred
    else:
        return get_predecessor_node(pred["node"])


def get_pred_node_by_gen_and_line(gen, line):
    pred = get_predecessor_node(f"{gen}_{line}")
    if pred is None:
        return None, None
    gen, line = pred["node"].split("_")
    return int(gen), int(line)


import polars as pl
from IPython.display import display
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


samples = client.query("select count(distinct(generation, line)) from methylome")
samples = samples.result_rows[0][0]

hist_mod_dict = {
    "input": 1,
    "H3": 2,
    "H3K4Me1": 3,
    "H3K27Me3": 4,
    "H2AZ": 5,
    "H3K56Ac": 6,
    "H3K4Me3": 7,
}


def save_surroundings(chromosome, location, idx, targets, mode="train"):
    idx = idx.tolist()
    targets = targets.tolist()
    data = [chromosome, location, idx, targets, mode]

    client.insert(
        "training_data",
        [data],
        column_names=["chromosome", "location", "prompt", "targets", "mode"],
    )


def get_neighbours(site, mode="train"):
    # get surrounding genes
    # tic = time.perf_counter()
    genes = client.query_arrow(
        f"select type, ({site['location']} - start) as start_diff, ({site['location']} - end) as end_diff from genes where chromosome = {site['chromosome']} and strand = {site['strand']} order by abs(start_diff) limit {gene_neighbours}"
    )
    g = pl.DataFrame(genes)

    ##toc = time.perf_counter()
    # print(f"genes in {#toc - #tic:0.4f} seconds")
    # display(g)
    # Get surrounding histone mods
    # tic = time.perf_counter()

    offset = 1000
    sufficient = False
    while not sufficient:
        histone_mods = client.query_arrow(
            f"select modification, ({site['location']} - start ) as start_diff, ({site['location']} - end) as end_diff from histone_mods where chromosome = {site['chromosome']} and start between {site['location'] - offset} and {site['location'] + offset} order by abs(start_diff) limit {hist_mod_neighbours}"
        )
        h = pl.DataFrame(histone_mods)
        if len(h) == hist_mod_neighbours:
            sufficient = True
        else:
            print(
                f"Only {len(h)} histone mods found, increasing offset to {offset * 2}"
            )
            offset *= 2
            continue

        h = h.with_columns(
            pl.col("modification").map_dict(hist_mod_dict).alias("modification")
        )

    # toc = time.perf_counter()
    # print(f"Histone mods in {#toc - #tic:0.4f} seconds")
    # display(h)
    # Get surrounding chromatine states
    # tic = time.perf_counter()
    chr_states = client.query_arrow(
        f"select state, ({site['location']} - start ) as start_diff, ({site['location']} -end) as end_diff from chr_states where chromosome = {site['chromosome']} order by abs(start_diff) limit {cs_neighbours } "
    )
    c = pl.DataFrame(chr_states)

    # toc = time.perf_counter()
    # print(f"Chromatine State in {#toc - #tic:0.4f} seconds")
    # display(c)
    # Get each site in all generations and lines
    # tic = time.perf_counter()
    offset = 200
    sufficient = False
    while not sufficient:
        all_generations_and_neighbours = client.query_arrow(
            f"select * except (trinucleotide_context, pedigree, id), ({site['location']} - location ) as location_diff from methylome where chromosome = {site['chromosome']} and strand = {site['strand']} and location between  {site['location'] - offset} and  {site['location'] + offset} order by abs(location_diff), location_diff, generation, line limit {(meth_neighbours +1)  * samples}"
        )
        m = pl.DataFrame(
            all_generations_and_neighbours
        )  # (meth_neighbours * samples, 12)
        if len(m) == (meth_neighbours + 1) * samples:
            sufficient = True
        else:
            # print(f"Only {len(m)} neighbours found, increasing offset to {offset * 2}")
            offset *= 2
            continue
    # display(m)

    # toc = time.perf_counter()
    # print(f"neighbours in {#toc - #tic:0.4f} seconds")
    # tic = time.perf_counter()
    site_across_generations = m.filter(pl.col("location_diff") == 0)
    m = m.filter(pl.col("location_diff") != 0)

    m = m.filter((pl.col("generation") != 0) & (pl.col("line") != 0))

    preds = []
    targets = []
    for site in site_across_generations.iter_rows(named=True):
        pred_gen, pred_line = get_pred_node_by_gen_and_line(
            site["generation"], site["line"]
        )
        if not pred_gen is None:
            pred = site_across_generations.filter(
                (pl.col("generation") == pred_gen) & (pl.col("line") == pred_line)
            )  # (1, 12)
            preds.append(torch.tensor(pred.to_numpy()[0], dtype=torch.float32))
            targets.append(site["meth_lvl"])

    # toc = time.perf_counter()
    # print(f"Filtering for {#toc - #tic:0.4f} seconds")
    # tic = time.perf_counter()
    t = torch.tensor(targets, dtype=torch.float32)
    p = torch.stack(preds)  # (samples - 1, 12)
    g = torch.tensor(g.to_numpy(), dtype=torch.float32)
    m = torch.tensor(m.to_numpy(), dtype=torch.float32).T
    h = torch.tensor(h.to_numpy(), dtype=torch.float32)
    c = torch.tensor(c.to_numpy(), dtype=torch.float32)

    m = m.reshape(
        (samples - 1), 12 * meth_neighbours
    )  # (samples -1, concatenated neighbours)

    # These are time-invariant
    g = g.reshape(3 * gene_neighbours)
    h = h.reshape(3 * hist_mod_neighbours)
    c = c.reshape(3 * cs_neighbours)

    # surroundings
    s = torch.cat([g, c, h])
    s = s.expand((samples - 1), -1)

    x = torch.cat([m, p, s], dim=1)

    # toc = time.perf_counter()
    # print(f"Reshaping in {#toc - #tic:0.4f} seconds")
    # tic = time.perf_counter()
    save_surroundings(site["chromosome"], site["location"], x, t, mode)
    # toc = time.perf_counter()
    # print(f"Saving in {#toc - #tic:0.4f} seconds")
    return x, t


import time
import numpy as np


def create_all_samples():
    start = 0
    end = 1e6
    total = 557173708

    while start < total:
        next_samples = client.query_arrow(
            f"select * except (trinucleotide_context, pedigree, id) from methylome where generation = 0 and context = 'CG' and location between {start} and {end} order by chromosome, location, strand"
        )
        df = pl.DataFrame(next_samples)

        # print(f"Creating {df.height} samples in range {start} to {end}")
        i = 0
        for site in df.iter_rows(named=True):
            mode = np.random.choice(["train", "test", "validation"], p=[0.8, 0.1, 0.1])
            # tic = time.perf_counter()
            # #print(
            #     f"Creating sample {i} in mode {mode} at location {site['location']} on chromosome {site['chromosome']}"
            # )
            get_neighbours(site, mode)
            # toc = time.perf_counter()
            # print(f"Created sample {i} in {toc - tic:0.4f} seconds")
            if i % 100 == 0:
                print(i)
            i += 1

        start = end
        end = end + 1e6


# tic = time.perf_counter()
create_all_samples()
# toc = time.perf_counter()
# print(f"Creating all samples in {#toc - #tic:0.4f} seconds")
