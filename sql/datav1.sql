----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS final_v1(
    id                      SERIAL          NOT NULL
    , "date"                DATE            NULL
    , store_nbr             INT             NULL
    , family                VARCHAR(40)     NULL
    , sales                 FLOAT           NULL
    , onpromotion           FLOAT           NULL
    , "dcoilwtico"          REAL            NULL
    , city                  VARCHAR(40)     NULL
    , "state"               VARCHAR(40)     NULL
    , "type"                CHAR(1)         NULL
    , cluster               INT             NULL
    , transactions          INT             NULL
    , "date_type_local"     VARCHAR(10)     NULL
    , "date_name_local"     VARCHAR(100)    NULL
    , "loc_local"           VARCHAR(100)    NULL
    , "date_type_region"    VARCHAR(10)     NULL
    , "date_name_region"    VARCHAR(100)    NULL
    , "loc_region"          VARCHAR(100)    NULL
    , "date_type_nation_1"  VARCHAR(10)     NULL
    , "date_name_nation_1"  VARCHAR(100)    NULL
    , "date_type_nation_2"  VARCHAR(10)     NULL
    , "date_name_nation_2"  VARCHAR(100)    NULL
    , "sales_last1"         FLOAT           NULL
    , "sales_last2"         FLOAT           NULL
    , "sales_last3"         FLOAT           NULL
    , "sales_last4"         FLOAT           NULL
);


----------------------------------------------------------------
INSERT INTO final_v1 (
    "date"
    , store_nbr
    , family
    , sales
    , onpromotion
    , "dcoilwtico"
    , city
    , "state"
    , "type"
    , cluster
    , transactions
    , "date_type_local"
    , "date_name_local"
    , "loc_local"
    , "date_type_region"
    , "date_name_region"
    , "loc_region"
    , "date_type_nation_1"
    , "date_name_nation_1"
    , "date_type_nation_2"
    , "date_name_nation_2"
    , "sales_last1"
    , "sales_last2"
    , "sales_last3"
    , "sales_last4"
)

SELECT
    f."date"con
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
    , f_last1.sales         AS  sales_last1
    , f_last2.sales         AS  sales_last2
    , f_last3.sales         AS  sales_last3
    , f_last4.sales         AS  sales_last4
FROM final f
    LEFT JOIN final f_last1 ON 1=1
        AND f_last1.store_nbr = f.store_nbr
        AND f_last1.family = f.family
        AND f.date - f_last1.date = 1
    LEFT JOIN final f_last2 ON 1=1
        AND f_last2.store_nbr = f.store_nbr
        AND f_last2.family = f.family
        AND f.date - f_last2.date = 2
    LEFT JOIN final f_last3 ON 1=1
        AND f_last3.store_nbr = f.store_nbr
        AND f_last3.family = f.family
        AND f.date - f_last3.date = 3
    LEFT JOIN final f_last4 ON 1=1
        AND f_last4.store_nbr = f.store_nbr
        AND f_last4.family = f.family
        AND f.date - f_last4.date = 4
    -- To deal with NULL in oil
    
;


WITH TMP_N_NULL AS (
    SELECT
        "date"
        , "family"
        , store_nbr
        , dcoilwtico
    FROM final_v1
    WHERE 1=1
        AND dcoilwtico IS NOT NULL
),
TMP_NULL AS (
    SELECT
        "date"
        , "family"
        , store_nbr
    FROM final_v1
    WHERE 1=1
        AND dcoilwtico IS NULL
)
    UPDATE final_v1 AS f
    SET
        dcoilwtico = f_sub.customer,
    FROM (
        SELECT 
            tmp_f."date"
            , tmp_f."family"
            , tmp_f.store_nbr
            , f_oil.dcoilwtico
        FROM TMP_NULL tmp_f
        LEFT JOIN LATERAL (
            SELECT FROM TMP_N_NULL tmp_f
            WHERE 1=1
                AND tmp_f.store_nbr = f.store_nbr
                AND tmp_f.family = f.family
                AND f.date - tmp_f.date BETWEEN 0 AND 10
                and f.dcoilwtico IS NOT NULL
            ORDER BY "date" DESC
            FETCH FIRST ROW ONLY
        ) AS f_oil ON true
    ) AS f_sub
    WHERE 1=1
        AND f."date" = f_sub."date"
        AND f."family" = f_sub."family"
        AND f.store_nbr = f_sub.store_nbr
;