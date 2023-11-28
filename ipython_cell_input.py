
import clickhouse_connect
def get_neighbours(site, mode="train"):
    # get surrounding genes
    client = clickhouse_connect.get_client(host='localhost', username='cgoeldel', password='Goe1409ldel')

    genes = client.query_arrow(f"select type, ({site['location']} - start) as start_diff, ({site['location']} - end) as end_diff from genes where chromosome = {site['chromosome']} and strand = {site['strand']} order by abs(start_diff) limit {gene_neighbours}")
    g = pl.DataFrame(genes)

    # display(g)
    # Get surrounding histone mods 
    histone_mods = client.query_arrow(f"select modification, ({site['location']} - start ) as start_diff, ({site['location']} - end) as end_diff from histone_mods where chromosome = {site['chromosome']}  order by abs(start_diff) limit {hist_mod_neighbours}")
    h = pl.DataFrame(histone_mods)
    h = h.with_columns(pl.col("modification").map_dict(hist_mod_dict).alias("modification"))
    # display(h)
    # Get surrounding chromatine states
    chr_states = client.query_arrow(f"select state, ({site['location']} - start ) as start_diff, ({site['location']} -end) as end_diff from chr_states where chromosome = {site['chromosome']} order by abs(start_diff) limit {cs_neighbours } ")
    c = pl.DataFrame(chr_states)

    # display(c)
    # Get each site in all generations and lines 
    all_generations_and_neighbours = client.query_arrow(f"select * except (trinucleotide_context, pedigree, id), ({site['location']} - location ) as location_diff from methylome where chromosome = {site['chromosome']} and strand = {site['strand']} order by abs(location_diff), location_diff, generation, line limit {(meth_neighbours +1)  * samples}")
    m = pl.DataFrame(all_generations_and_neighbours) # (meth_neighbours * samples, 12)
    # display(m)

    site_across_generations = m.filter(pl.col("location_diff") == 0)
    m = m.filter(pl.col("location_diff") != 0)
   
    m = m.filter((pl.col("generation") != 0) & (pl.col("line") != 0))

    preds = []
    targets = []
    for site in site_across_generations.iter_rows(named=True):
        pred_gen, pred_line = get_pred_node_by_gen_and_line(site["generation"], site["line"])
        if not pred_gen is None:
            pred = site_across_generations.filter((pl.col("generation") == pred_gen) & (pl.col("line") == pred_line)) # (1, 12)
            preds.append(torch.tensor(pred.to_numpy()[0], dtype=torch.float32))
            targets.append(site["meth_lvl"])

    t = torch.tensor(targets, dtype=torch.float32)
    p = torch.stack(preds) # (samples - 1, 12)
    g = torch.tensor(g.to_numpy(), dtype=torch.float32) 
    m = torch.tensor(m.to_numpy(),dtype=torch.float32).T 
    h = torch.tensor(h.to_numpy(),dtype=torch.float32)
    c = torch.tensor(c.to_numpy(),dtype=torch.float32)

    m = m.reshape((samples-1), 12 * meth_neighbours) # (samples -1, concatenated neighbours)
    
    # These are time-invariant
    g = g.reshape(3 * gene_neighbours)
    h = h.reshape(3 * hist_mod_neighbours)
    c = c.reshape(3* cs_neighbours)

    # surroundings 
    s = torch.cat([g, c, h])
    s = s.expand((samples -1), -1)

    x = torch.cat([m, p, s], dim=1)

    save_surroundings(site["chromosome"], site["location"], x, t, mode)

    return x, t

get_neighbours(100)
