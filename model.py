import torch
import clickhouse_connect
import polars as pl
import time

import torch

max_sites_per_gene = 512
max_chr_states_per_gene = 512  # including gene itself
num_genes = 18972

device = "cuda" if torch.cuda.is_available() else "cpu"


class Sites(torch.utils.data.IterableDataset):
    def __init__(self, mode="train"):
        super(Sites).__init__()
        self.mode = mode

        self.client = clickhouse_connect.get_client(
            host="localhost", username="cgoeldel", password="Goe1409ldel"
        )
        genes = (
            self.client.query_arrow(
                f"select * from genes where chromosome < 5 and type = 'gbM'"
            )
            if mode == "train"
            else self.client.query_arrow(
                f"select * from genes where chromosome = 5 and type = 'gbM'"
            )
        )
        self.genes = pl.DataFrame(genes)

        # self.samples_per_chrom = pl.DataFrame(self.client.query_arrow("select chromosome, count(*) from methylome_within_gbM where generation = 0 group by chromosome").result_rows[0][0])
        # self.samples = self.client.query("select count(distinct(generation, line)) from methylome_within_gbM").results_rows[0][0]
        # self.hist_mod_dict = {
        #         "input" : 1,
        #         "H3" : 2,
        #         "H3K4Me1" : 3,
        #         "H3K27Me3" : 4,
        #         "H2AZ" : 5,
        #         "H3K56Ac" : 6,
        #         "H3K4Me3" : 7
        #         }

    def __iter__(self):
        for gene in self.genes.iter_rows(named=True):
            tic = time.perf_counter()

            # TODO: more elegant way to iterate over all pedigree branches
            sites_line_2 = self.client.query_arrow(
                f"select location - {gene['start']} as start_diff, {gene['end']} -location as end_diff, start_diff * 1.0 / ({gene['end'] - gene['start']}) as percentile, meth_lvl from methylome_within_gbM where chromosome = {gene['chromosome']} and (line = 2 or line = 0)  and strand = {gene['strand']} and location between {gene['start']} and {gene['end']} order by generation, line, start_diff"
            )
            num_sites = len(sites_line_2)
            num_generations = 7
            sites_line_2 = pl.DataFrame(sites_line_2)["meth_lvl"].to_numpy()
            # sites_line_8 = self.client.query_arrow(f"select location - {gene['start']} as start_diff, {gene['end']} -location as end_diff, location * 1.0 / ({gene_length}) as percentile, meth_lvl from methylome_within_gbM where chromosome = {gene['chromosome']} and (line = 8 or line = 0)  and strand = {gene['strand']} and location between {gene['start']} and {gene['end']} order by generation, line, start_diff")
            # sites_line_8 = pl.DataFrame(sites_line_8)

            # targets_line_8 = sites_line_8['meth_lvl'].to_numpy()

            # targets_line_8 = torch.tensor(targets_line_8, dtype=torch.float32)

            # targets_line_8 = targets_line_8.reshape(gene_length, 6)

            # remove first row, these can not be predicted as there is no predecessor
            # targets_line_8 = targets_line_8[1:, :]
            sites_line_2 = torch.tensor(sites_line_2, dtype=torch.float32).reshape(
                num_sites // num_generations, num_generations
            )
            targets_line_2 = sites_line_2[1:, :]
            # sites_line_8 = torch.tensor(sites_line_8.to_numpy(), dtype=torch.float32).reshape(gene_length, 6)

            # remove the last row, these have no predecessor, so no known target
            sites_line_2 = sites_line_2[:-1, :]
            # sites_line_8 = sites_line_8[:-1, :]

            chr_states = self.client.query_arrow(
                f"select chromosome, (start - {gene['start']} ) as start_diff, ({gene['end']} - end) as end_diff, state from chr_states where chromosome = {gene['chromosome']} and start between {gene['start']} and {gene['end']} and end between {gene['start']} and {gene['end']}"
            )
            chr_states = pl.DataFrame(chr_states)
            gene_and_divider = pl.DataFrame(
                {
                    "chromosome": [gene["chromosome"]],
                    "start_diff": [gene["start"]],
                    "end_diff": [gene["end"]],
                    "state": [255],
                },
                schema={
                    "chromosome": pl.UInt8,
                    "start_diff": pl.Int64,
                    "end_diff": pl.Int64,
                    "state": pl.UInt8,
                },
            )
            chr_states_and_gene = pl.concat(
                [gene_and_divider, chr_states], how="vertical"
            )

            chr_states = torch.tensor(
                chr_states_and_gene.to_numpy(), dtype=torch.float32
            ).flatten()

            # print(sites_line_2.shape)
            # print(sites_line_8.shape)
            # print(targets_line_2.shape)
            # print(targets_line_8.shape)
            # print(chr_states.shape)

            x = torch.zeros(max_sites_per_gene, num_generations)
            t = torch.zeros(max_sites_per_gene, num_generations)
            c = torch.zeros(max_chr_states_per_gene, num_generations)

            x[: sites_line_2.shape[0], :] = sites_line_2
            t[: targets_line_2.shape[0], :] = targets_line_2
            c[: chr_states.shape[0]] = chr_states.repeat(num_generations, 1).T

            # x = torch.unsqueeze(x.T, 0)
            # t = torch.unsqueeze(t.T, 0)
            # c = torch.unsqueeze(c.T, 0)
            # print(x.shape, t.shape, c.shape)

            # every last node -> list back to root: decoder
            # surrounding methylome -> encoder
            # surrounding gene + chromatine state -> encoder
            x = x.T.to(device)
            c = c.T.to(device)
            t = t.T.to(device)

            toc = time.perf_counter()
            # print(
            #     f"Gene {gene['chromosome']}:{gene['start']}:{gene['end']} in {toc - tic:0.4f} seconds"
            # )
            yield x, c, t


from torch import nn
from torch.nn import functional as F


class MethylationMaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=512,
            nhead=1,
            dim_feedforward=512 * 4,
            dtype=torch.float32,
            # batch_first=True,
        )

    def forward(self, x, c, targets=None):
        logits = self.transformer(
            c,
            x,
            tgt_is_causal=True,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(
                device
            ),
        )

        if targets is None:
            return logits, None
        # print(logits, targets)
        loss = F.mse_loss(logits, targets)

        return logits, loss


from torch.utils.data import DataLoader

training_data = Sites(mode="train")
test_data = Sites(mode="test")
train_dataloader = DataLoader(training_data, batch_size=None)
test_dataloader = DataLoader(test_data, batch_size=None)
conschti = MethylationMaster().to(device)

print(sum(p.numel() for p in conschti.parameters()) / 1e6, "M parameters")
optimizer = torch.optim.Adam(conschti.parameters(), lr=1e-3)


genes_per_chrom = {
    1: 4952,
    2: 3003,
    3: 3721,
    4: 2864,
    5: 4432,
}

training_genes = sum(genes_per_chrom.values()) - genes_per_chrom[5]
test_genes = genes_per_chrom[5]

run_name = "batched_by_generation_causal"
with open(f"log/{run_name}.txt", "w") as f:
    # reset log file
    f.write("step,time,loss,test_loss\n")

tic = time.perf_counter()
for i in range(training_genes):
    optimizer.zero_grad()

    x, c, t = next(iter(train_dataloader))

    logits, loss = conschti(x, c, t)
    loss.backward()
    optimizer.step()

    if i % training_genes // test_genes == 0:
        x, c, t = next(iter(test_dataloader))
        logits, test_loss = conschti(x, c, t)
        toc = time.perf_counter()
        # write step, timing, loss to file log.txt
        with open(f"log/{run_name}.txt", "a") as f:
            f.write(f"{i},{toc - tic:0.4f},{loss},{test_loss}\n")
