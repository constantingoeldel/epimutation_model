select * except (trinucleotide_context, pedigree, id)
from methylome sample {sample} offset {offset} 
where generation = 0 order by rand() limit {batch_size}

select type, ({site['location']} - start) as start_diff, ({site['location']} - end) as end_diff 
from genes where chromosome = {site['chromosome']} and strand = {site['strand']} 
order by abs(start_diff) limit {gene_neighbours}

select modification, ({site['location']} - start ) as start_diff, ({site['location']} - end) as end_diff 
from histone_mods where chromosome = {site['chromosome']}  
order by abs(start_diff) limit {hist_mod_neighbours}

select state, ({site['location']} - start ) as start_diff, ({site['location']} -end) as end_diff 
from chr_states where chromosome = {site['chromosome']} 
order by abs(start_diff) limit {cs_neighbours }

select * except (trinucleotide_context, pedigree, id), ({site['location']} - location ) as location_diff 
from methylome where chromosome = {site['chromosome']} and strand = {site['strand']} 
order by abs(location_diff), location_diff, generation, line limit {(meth_neighbours +1)  * samples}