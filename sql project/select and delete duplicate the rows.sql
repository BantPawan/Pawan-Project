-- select duplicates 
SELECT transaction_id 
     transaction_date, 
      transaction_time, 
    COUNT(*) AS CNT
FROM power_bi_exp.coffee_shop_sales
GROUP BY transaction_id, 
     transaction_date, 
      transaction_time
HAVING COUNT(*) > 1;
USE power_bi_exp;

-- delete the duplicates

WITH CTE AS (
    SELECT transaction_id, 
           transaction_date, 
           transaction_time,
           ROW_NUMBER() OVER(PARTITION BY transaction_id, transaction_date, transaction_time ORDER BY transaction_id) AS DuplicateCount
    FROM coffee_shop_sales
)
DELETE FROM coffee_shop_sales
WHERE transaction_id IN (
    SELECT transaction_id
    FROM CTE
    WHERE DuplicateCount > 1
);
