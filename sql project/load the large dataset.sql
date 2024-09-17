use power_bi_exp;

LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/coffee_shop_sales_chunk_15.csv"
INTO TABLE coffee_shop_sales_14
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SHOW VARIABLES LIKE 'secure_file_priv';

create table coffee_shop_sales_14
(
	transaction_id int,
transaction_date text,
transaction_time text,
transaction_qty int,
store_id int,
store_location text,
product_id int,
unit_price double,
product_category text,
product_type text,
product_detail text
);

use powerbi_project;


CREATE TABLE coffee_shop_sales AS
SELECT * FROM power_bi_exp.coffee_shop_sales WHERE 1=0;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_1;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_2;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_3;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_4;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_5;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_6;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_7;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_8;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_9;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_10;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_11;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_12;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_13;

INSERT INTO coffee_shop_sales
SELECT * FROM power_bi_exp.coffee_shop_sales_14;
