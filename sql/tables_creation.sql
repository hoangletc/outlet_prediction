CREATE TABLE IF NOT EXISTS transactions(
    id              SERIAL      PRIMARY KEY
    , "date"        DATE        NOT NULL
    , store_nbr     INT         NOT NULL
    , family        VARCHAR(20) NOT NULL
    , sales         FLOAT       NOT NULL
    , onpromotion   FLOAT       NOT NULL
);

-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS oilprice(
    ID              SERIAL  PRIMARY KEY
    , "date"        DATE    NOT NULL
    , "dcoilwtico"   REAL    NOT NULL
);

-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS stores(
    id              SERIAL      PRIMARY KEY
    , store_nbr     INT         NOT NULL
    , city          VARCHAR(20) NOT NULL
    , "state"       VARCHAR(20) NOT NULL
    , "type"        CHAR(1)     NOT NULL
    , cluster       INT         NOT NULL
);

-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS total_trans(
    id              SERIAL      PRIMARY KEY
    , "date"        DATE        NOT NULL
    , store_nbr     INT         NOT NULL
    , transactions  INT         NOT NULL
);

-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS holidays(
    id              SERIAL          PRIMARY KEY
    , "date"        DATE            NOT NULL
    , "date_type"   VARCHAR(10)     NOT NULL
    , "date_name"   VARCHAR(100)    NOT NULL
    , "locale_name" VARCHAR(100)    NOT NULL
);

-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS final(
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
);

