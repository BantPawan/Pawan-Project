# ☕ Coffee Shop Sales Analysis (SQL)

This repository contains SQL queries for data cleaning, KPI calculations, trend analysis, and sales insights. The goal is to analyze transactions and derive meaningful business metrics.


---

## 📌 Table of Contents

## 📊 Data Cleaning & Formatting

## 📈 KPI Calculations & Sales Analysis

## 📅 Sales Trends & Comparisons

## 📊 Category-Wise Sales Insights



---

## 📊 Data Cleaning & Formatting

-- Convert `transaction_date` column to proper date format
UPDATE coffee_shop_sales
SET transaction_date = STR_TO_DATE(transaction_date, '%d-%m-%Y');

-- Alter `transaction_date` column to date data type
ALTER TABLE coffee_shop_sales
MODIFY COLUMN transaction_date DATE;

-- Convert `transaction_time` column to proper time format
UPDATE coffee_shop_sales
SET transaction_time = STR_TO_DATE(transaction_time, '%H:%i:%s');

-- Alter `transaction_time` column to time data type
ALTER TABLE coffee_shop_sales
MODIFY COLUMN transaction_time TIME;

-- Check data types of different columns
DESCRIBE coffee_shop_sales;


---

## 📈 KPI Calculations & Sales Analysis

### Total Sales for May

SELECT ROUND(SUM(unit_price * transaction_qty)) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5;

### Total Sales Month-on-Month (MoM) Growth

SELECT 
    MONTH(transaction_date) AS month,
    ROUND(SUM(unit_price * transaction_qty)) AS total_sales,
    (SUM(unit_price * transaction_qty) - 
        LAG(SUM(unit_price * transaction_qty), 1) OVER (ORDER BY MONTH(transaction_date))) / 
        LAG(SUM(unit_price * transaction_qty), 1) OVER (ORDER BY MONTH(transaction_date)) * 100 
    AS mom_increase_percentage
FROM coffee_shop_sales
WHERE MONTH(transaction_date) IN (4, 5)
GROUP BY MONTH(transaction_date)
ORDER BY MONTH(transaction_date);

### Total Orders for May

SELECT COUNT(transaction_id) AS Total_Orders
FROM coffee_shop_sales 
WHERE MONTH(transaction_date) = 5;

### Total Orders MoM Growth

SELECT 
    MONTH(transaction_date) AS month,
    ROUND(COUNT(transaction_id)) AS total_orders,
    (COUNT(transaction_id) - 
        LAG(COUNT(transaction_id), 1) OVER (ORDER BY MONTH(transaction_date))) / 
        LAG(COUNT(transaction_id), 1) OVER (ORDER BY MONTH(transaction_date)) * 100 
    AS mom_increase_percentage
FROM coffee_shop_sales
WHERE MONTH(transaction_date) IN (4, 5)
GROUP BY MONTH(transaction_date)
ORDER BY MONTH(transaction_date);

### Total Quantity Sold for May

SELECT SUM(transaction_qty) AS Total_Quantity_Sold
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5;

### Total Quantity Sold MoM Growth

SELECT 
    MONTH(transaction_date) AS month,
    ROUND(SUM(transaction_qty)) AS total_quantity_sold,
    (SUM(transaction_qty) - 
        LAG(SUM(transaction_qty), 1) OVER (ORDER BY MONTH(transaction_date))) / 
        LAG(SUM(transaction_qty), 1) OVER (ORDER BY MONTH(transaction_date)) * 100 
    AS mom_increase_percentage
FROM coffee_shop_sales
WHERE MONTH(transaction_date) IN (4, 5)
GROUP BY MONTH(transaction_date)
ORDER BY MONTH(transaction_date);


---

## 📅 Sales Trends & Comparisons

### Daily Sales, Quantity, and Orders for a Specific Date

SELECT 
    SUM(unit_price * transaction_qty) AS total_sales,
    SUM(transaction_qty) AS total_quantity_sold,
    COUNT(transaction_id) AS total_orders
FROM coffee_shop_sales
WHERE transaction_date = '2023-05-18';

## Average Daily Sales in May

SELECT AVG(total_sales) AS average_sales
FROM (
    SELECT SUM(unit_price * transaction_qty) AS total_sales
    FROM coffee_shop_sales
    WHERE MONTH(transaction_date) = 5
    GROUP BY transaction_date
) AS internal_query;

### Daily Sales Breakdown for May

SELECT 
    DAY(transaction_date) AS day_of_month,
    ROUND(SUM(unit_price * transaction_qty), 1) AS total_sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY DAY(transaction_date)
ORDER BY DAY(transaction_date);

### Daily Sales vs Average Comparison

SELECT 
    day_of_month,
    CASE 
        WHEN total_sales > avg_sales THEN 'Above Average'
        WHEN total_sales < avg_sales THEN 'Below Average'
        ELSE 'Average'
    END AS sales_status,
    total_sales
FROM (
    SELECT 
        DAY(transaction_date) AS day_of_month,
        SUM(unit_price * transaction_qty) AS total_sales,
        AVG(SUM(unit_price * transaction_qty)) OVER () AS avg_sales
    FROM coffee_shop_sales
    WHERE MONTH(transaction_date) = 5
    GROUP BY DAY(transaction_date)
) AS sales_data
ORDER BY day_of_month;

### Weekday vs Weekend Sales

SELECT 
    CASE 
        WHEN DAYOFWEEK(transaction_date) IN (1, 7) THEN 'Weekends'
        ELSE 'Weekdays'
    END AS day_type,
    ROUND(SUM(unit_price * transaction_qty), 2) AS total_sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY day_type;

### Sales by Hour of the Day

SELECT 
    HOUR(transaction_time) AS Hour_of_Day,
    ROUND(SUM(unit_price * transaction_qty)) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY HOUR(transaction_time)
ORDER BY HOUR(transaction_time);

### Sales by Day of the Week

SELECT 
    CASE 
        WHEN DAYOFWEEK(transaction_date) = 2 THEN 'Monday'
        WHEN DAYOFWEEK(transaction_date) = 3 THEN 'Tuesday'
        WHEN DAYOFWEEK(transaction_date) = 4 THEN 'Wednesday'
        WHEN DAYOFWEEK(transaction_date) = 5 THEN 'Thursday'
        WHEN DAYOFWEEK(transaction_date) = 6 THEN 'Friday'
        WHEN DAYOFWEEK(transaction_date) = 7 THEN 'Saturday'
        ELSE 'Sunday'
    END AS Day_of_Week,
    ROUND(SUM(unit_price * transaction_qty)) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY Day_of_Week;


---

## 📊 Category-Wise Sales Insights

### Sales by Store Location

SELECT store_location, 
    SUM(unit_price * transaction_qty) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY store_location
ORDER BY Total_Sales DESC;

### Sales by Product Category

SELECT product_category, 
    ROUND(SUM(unit_price * transaction_qty), 1) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY product_category
ORDER BY Total_Sales DESC;

### Top 10 Products by Sales

SELECT product_type, 
    ROUND(SUM(unit_price * transaction_qty), 1) AS Total_Sales
FROM coffee_shop_sales
WHERE MONTH(transaction_date) = 5
GROUP BY product_type
ORDER BY Total_Sales DESC
LIMIT 10;

### Sales for Tuesday at 8 AM

SELECT 
    ROUND(SUM(unit_price * transaction_qty)) AS Total_Sales,
    SUM(transaction_qty) AS Total_Quantity,
    COUNT(*) AS Total_Orders
FROM coffee_shop_sales
WHERE DAYOFWEEK(transaction_date) = 3  -- Tuesday
    AND HOUR(transaction_time) = 8     -- 8 AM
    AND MONTH(transaction_date) = 5;   -- May
