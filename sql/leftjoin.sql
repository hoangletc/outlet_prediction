WITH TMP AS (
    SELECT
        t.*
        , s.type
        , s.city
        , s.state
        , s.cluster
        , tt.transactions
        , o.dcoilwtico

        , CASE WHEN h_local.id IS NULL 
                OR h_nation1.date_name IN ('navidad', 'primer dia del ano')
                THEN 'ignored'
            ELSE h_local.date_type
        END                                         AS date_type_local
        , CASE WHEN h_local.id IS NULL
                OR h_nation1.date_name IN ('navidad', 'primer dia del ano')
                THEN 'ignored'
            ELSE h_local.date_name
        END                                         AS date_name_local

        , CASE WHEN h_region.id IS NULL
                OR h_nation1.date_name IN ('navidad', 'primer dia del ano')
                THEN 'ignored'
            ELSE h_region.date_type
        END                                         AS date_type_region
        , CASE WHEN h_region.id IS NULL
                OR h_nation1.date_name IN ('navidad', 'primer dia del ano')
                THEN 'ignored'
            ELSE h_region.date_name
        END                                         AS date_name_region

        , CASE WHEN h_nation1.id IS NULL THEN (
                CASE WHEN extract(isodow from t.date) - 1 >= 5
                    THEN 'weekend'
                    ELSE 'work day'
                END)
            ELSE h_nation1.date_type
        END                                         AS date_type_nation_1
        , CASE WHEN h_nation1.id IS NULL THEN (
                CASE WHEN extract(isodow from t.date) - 1 >= 5
                    THEN 'weekend'
                    ELSE 'work day'
                END)
            ELSE h_nation1.date_name
        END                                         AS date_name_nation_1

        , CASE WHEN h_nation2.id IS NULL OR h_nation1.id = h_nation2.id 
                THEN 'ignored'
            ELSE h_nation2.date_type
        END                                         AS date_type_nation_2
        , CASE WHEN h_nation2.id IS NULL OR h_nation1.id = h_nation2.id 
                THEN 'ignored'
            ELSE h_nation2.date_name
        END                                         AS date_name_nation_2
    FROM transactions t
        LEFT JOIN stores s ON 1=1
            AND s.store_nbr = t.store_nbr
        LEFT JOIN total_trans tt ON 1=1
            AND tt.date = t.date
            AND tt.store_nbr = t.store_nbr
        LEFT JOIN oilprice o ON 1=1
            AND o.date = t.date
        
        LEFT JOIN holidays h_local ON 1=1
            AND s.city = h_local.locale_name
            AND h_local.date = t.date
        LEFT JOIN holidays h_region ON 1=1
            AND s.state = h_region.locale_name
            AND h_region.date = t.date
        LEFT JOIN LATERAL (
            SELECT *
            FROM holidays
            WHERE 1=1
                AND holidays.locale_name = 'ecuador'
                AND holidays.date = t.date
            ORDER BY id ASC
            FETCH FIRST ROW ONLY
        ) AS h_nation1 ON true
        LEFT JOIN LATERAL (
            SELECT *
            FROM holidays
            WHERE 1=1
                AND holidays.locale_name = 'ecuador'
                AND holidays.date = t.date
            ORDER BY id DESC
            FETCH FIRST ROW ONLY
        ) AS h_nation2 ON true
)
    INSERT INTO final(
        "date"
        , store_nbr
        , sales
        , onpromotion
        , dcoilwtico
        , cluster
        , transactions
        , "state"
        , "type"
        , date_type_nation_2
        , date_name_nation_2
        , date_type_local
        , date_name_local
        , loc_local
        , date_type_region
        , date_name_region
        , family
        , loc_region
        , date_type_nation_1
        , date_name_nation_1
        , city
    )
    select
        "date"
        , store_nbr
        , sales
        , onpromotion
        , dcoilwtico
        , cluster
        , transactions
        , "state"
        , "type"
        , date_type_nation_2
        , date_name_nation_2
        , date_type_local
        , date_name_local
        , loc_local
        , date_type_region
        , date_name_region
        , family
        , loc_region
        , date_type_nation_1
        , date_name_nation_1
        , city
    FROM TMP
;

-- select * from final_trans 
-- where 1=1
-- -- ORDER BY id
-- limit 200 
-- ;

-- SELECT column_name
-- FROM information_schema.columns
-- where table_name = 'final';


-- select * from holidays
-- WHERE locale_name = 'ecuador'
--  order by date
--  limit 50;



-- select * from total_trans limit 10;