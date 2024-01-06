with percentile_windows as (
    select Count(*) as sites,
        FLOOR(
            (
                CASE
                    WHEN a.strand = '+' then (c.start - a.start)
                    else (a.end - c."end"
                )
            end * 100
        ) / (a.end - a.start
)
) as percentile
from histone_mods c
    JOIN genes a ON c.chromosome = a.chromosome
    and c.chromosome = 1
    and a.chromosome = 1
    and a.type = 'UM'
    and c.modification = 'H2AZ'
where c.start + (a.end - a.start
) >= a.start
and c."end" - (a.end - a.start
) <= a."end"
group by percentile
)
select sites,
    SUM(sites) OVER (
        ORDER BY percentile ROWS BETWEEN CURRENT ROW
            AND 4 FOLLOWING
    ) AS sites_sliding_sum,
    percentile
from percentile_windows