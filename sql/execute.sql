with TMP AS (
    SELECT
        f."date"
        , f.store_nbr
        , f.family
        , f.sales
        , f.onpromotion
        , COALESCE(f."dcoilwtico", f_oil.dcoilwtico)
        , f.city
        , f."state"
        , f."type"
        , f.cluster
        , f.transactions
        , f."date_type_local"
        , f."date_name_local"
        , f."loc_local"
        , f."date_type_region"
        , f."date_name_region"
        , f."loc_region"
        , f."date_type_nation_1"
        , f."date_name_nation_1"
        , f."date_type_nation_2"
        , f."date_name_nation_2"
    --     , f_last1.sales         AS  sales_last1
    --     , f_last2.sales         AS  sales_last2
    --     , f_last3.sales         AS  sales_last3
    --     , f_last4.sales         AS  sales_last4
    FROM final f
    WHERE f.date <= '2013-01-10'
)
--     LEFT JOIN final f_last1 ON 1=1
--         AND f_last1.store_nbr = f.store_nbr
--         AND f_last1.family = f.family
--         AND f.date - f_last1.date = 1
--     LEFT JOIN final f_last2 ON 1=1
--         AND f_last2.store_nbr = f.store_nbr
--         AND f_last2.family = f.family
--         AND f.date - f_last2.date = 2
--     LEFT JOIN final f_last3 ON 1=1
--         AND f_last3.store_nbr = f.store_nbr
--         AND f_last3.family = f.family
--         AND f.date - f_last3.date = 3
--     LEFT JOIN final f_last4 ON 1=1
--         AND f_last4.store_nbr = f.store_nbr
--         AND f_last4.family = f.family
--         AND f.date - f_last4.date = 4
    -- To deal with NULL in oil
    SELECT
        "date"
        , store_nbr
        , family
        , f.dcoilwtico
        , f_oil.dcoilwtico as "value"
    FROM TMP f
    LEFT JOIN LATERAL (
        SELECT dcoilwtico FROM TMP tmp_f
        WHERE 1=1
            AND tmp_f.store_nbr = f.store_nbr
            AND tmp_f.family = f.family
            AND tmp_f.date - f.date BETWEEN 1 AND 7
            and f.dcoilwtico IS NOT NULL
        ORDER BY "date" DESC
        FETCH FIRST ROW ONLY
    ) AS f_oil ON true
;

select * from final limit 10;