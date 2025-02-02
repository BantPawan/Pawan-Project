# ☕ Coffee Shop Sales Analysis (SQL)  

This repository contains SQL queries for **data cleaning, KPI calculations, trend analysis, and sales insights**. The goal is to analyze transactions and derive meaningful business metrics.  

---

## 📌 Table of Contents  
- [📊 Data Cleaning & Formatting](#-data-cleaning--formatting)  
- [📈 KPI Calculations & Sales Analysis](#-kpi-calculations--sales-analysis)  
- [📅 Sales Trends & Comparisons](#-sales-trends--comparisons)  
- [📊 Category-Wise Sales Insights](#-category-wise-sales-insights)  
- [📜 License](#-license)  

---

## 📊 Data Cleaning & Formatting  

Before performing any analysis, we ensure that the **date** and **time** columns are in the correct format.  

```sql
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
