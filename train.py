import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import SimpleTokenizer, TransformerIntentClassifier, MAX_LEN

MODEL_PATH = "intent_model.pth"

# =====================================================
# TRAINING DATA
# =====================================================

intent_data = [

# =====================================================
# SINGLE INTENTS
# =====================================================

# Upload File
("upload a file", ["upload_file"]),
("import a csv file", ["upload_file"]),
("load my dataset", ["upload_file"]),
("add an excel sheet", ["upload_file"]),
("bring in the spreadsheet", ["upload_file"]),

# Create Sheet
("create a new sheet", ["create_new_sheet"]),
("add a worksheet", ["create_new_sheet"]),
("start a fresh sheet", ["create_new_sheet"]),
("make another sheet", ["create_new_sheet"]),

# Name Sheet
("name the sheet sales report", ["name_new_sheet"]),
("call this sheet revenue data", ["name_new_sheet"]),
("rename the sheet", ["name_new_sheet"]),
("set sheet name to report", ["name_new_sheet"]),

# Base Sheet
("set this as base sheet", ["set_base_sheet"]),
("make this the base sheet", ["set_base_sheet"]),
("use this sheet as the base", ["set_base_sheet"]),

# Preprocessing Sheet
("set preprocessing sheet name", ["set_pre_sheet_name"]),
("define pre sheet as cleaned data", ["set_pre_sheet_name"]),
("use cleaned data as preprocessing", ["set_pre_sheet_name"]),

# Postprocessing Sheet
("set postprocessing sheet name", ["set_post_sheet_name"]),
("define post sheet as final output", ["set_post_sheet_name"]),
("use final output for postprocessing", ["set_post_sheet_name"]),

# Rename Column
("rename this column", ["set_new_column_name"]),
("change the column name", ["set_new_column_name"]),
("set column name to revenue", ["set_new_column_name"]),
("update column header to profit", ["set_new_column_name"]),

# X Axis
("set x axis", ["set_x_axis"]),
("use date as x axis", ["set_x_axis"]),
("make month the x axis", ["set_x_axis"]),
("assign quarter to x axis", ["set_x_axis"]),

# Y Axis
("set y axis", ["set_y_axis"]),
("use revenue as y axis", ["set_y_axis"]),
("make profit the y axis", ["set_y_axis"]),
("assign growth to y axis", ["set_y_axis"]),

# Open Column Builder
("open column builder", ["open_column_builder"]),
("launch column builder", ["open_column_builder"]),
("go to column builder", ["open_column_builder"]),
("edit columns", ["open_column_builder"]),


# =====================================================
# DOUBLE INTENTS
# =====================================================

("upload a file and create a sheet",
 ["upload_file", "create_new_sheet"]),

("create a sheet and name it report",
 ["create_new_sheet", "name_new_sheet"]),

("upload file and rename a column",
 ["upload_file", "set_new_column_name"]),

("open column builder and rename column",
 ["open_column_builder", "set_new_column_name"]),

("set x axis to date and y axis to revenue",
 ["set_x_axis", "set_y_axis"]),

("create sheet and set it as base",
 ["create_new_sheet", "set_base_sheet"]),

("rename column and set y axis to profit",
 ["set_new_column_name", "set_y_axis"]),

("set preprocessing and postprocessing sheets",
 ["set_pre_sheet_name", "set_post_sheet_name"]),


# =====================================================
# TRIPLE INTENTS
# =====================================================

("upload file, create sheet, and name it sales",
 ["upload_file", "create_new_sheet", "name_new_sheet"]),

("create sheet, name it report, and set as base",
 ["create_new_sheet", "name_new_sheet", "set_base_sheet"]),

("upload file, rename column, and open column builder",
 ["upload_file", "set_new_column_name", "open_column_builder"]),

("set x axis to month, y axis to revenue, and rename column",
 ["set_x_axis", "set_y_axis", "set_new_column_name"]),

("create sheet, set preprocessing sheet, and set postprocessing sheet",
 ["create_new_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),


# =====================================================
# FULL WORKFLOW (ALL INTENTS PRESENT)
# =====================================================

("upload the spreadsheet, create a sheet called sales dashboard, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, rename the column to revenue, set month as x axis and profit as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),

("import my dataset, add a worksheet named performance report, make it the base, assign cleaned data for preprocessing and final output for postprocessing, change column name to total sales, use date as x axis and growth as y axis, then launch column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),

("load the csv file, create and name a sheet revenue insights, select it as base sheet, configure preprocessing as cleaned data and postprocessing as final output, update column header to margin, assign quarter to x axis and earnings to y axis, and open the column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),

("bring in the excel file, start a fresh sheet called annual summary, make it the base sheet, set pre sheet to data prep and post sheet to final metrics, rename column to net revenue, configure year as x axis and profit as y axis, and go to column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
("add a formula column", ["add_formula_column"]),
("create a calculated column", ["add_formula_column"]),
("make a new formula column", ["add_formula_column"]),
("add a derived column", ["add_formula_column"]),
("create a computed field", ["add_formula_column"]),
("insert a formula-based column", ["add_formula_column"]),
("add a column using a formula", ["add_formula_column"]),
("build a formula column", ["add_formula_column"]),
("open column builder and add a formula column",
 ["open_column_builder", "add_formula_column"]),

("create a sheet and add a calculated column",
 ["create_new_sheet", "add_formula_column"]),

("upload file and create a formula column",
 ["upload_file", "add_formula_column"]),
("upload file, open column builder, and add a formula column",
 ["upload_file", "open_column_builder", "add_formula_column"]),

("create sheet, rename column, and add a calculated column",
 ["create_new_sheet", "set_new_column_name", "add_formula_column"]),
("upload the file, create a new sheet called financial dashboard, set it as base sheet, configure preprocessing and postprocessing sheets, add a formula column, rename it to profit margin, set date as x axis and revenue as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
("add revenue plus profit", ["plus"]),
("calculate revenue plus expenses", ["plus"]),
("sum profit plus tax", ["plus"]),
("add sales plus growth", ["plus"]),
("revenue plus margin", ["plus"]),
("profit plus cost", ["plus"]),
("growth plus revenue", ["plus"]),
("add column a plus column b", ["plus"]),
("compute revenue plus profit", ["plus"]),
("total sales plus tax", ["plus"]),
    ("add a formula column with revenue plus profit",
 ["add_formula_column", "plus"]),

("create a calculated column using sales plus tax",
 ["add_formula_column", "plus"]),

("rename column and calculate revenue plus growth",
 ["set_new_column_name", "plus"]),

("open column builder and compute profit plus margin",
 ["open_column_builder", "plus"]),
("upload the file, create a new sheet called combined report, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, add a formula column using revenue plus profit, rename this column to total revenue, set month as x axis and growth as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "plus",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
# =====================================================
# MINUS (Subtraction Intent)
# =====================================================

("subtract revenue minus expenses", ["minus"]),
("calculate profit minus tax", ["minus"]),
("revenue minus cost", ["minus"]),
("sales minus returns", ["minus"]),
("growth minus decline", ["minus"]),
("earnings minus expenses", ["minus"]),
("margin minus discount", ["minus"]),
("net income minus tax", ["minus"]),
("subtract column a minus column b", ["minus"]),
("compute revenue minus profit", ["minus"]),
("add a formula column with revenue minus expenses",
 ["add_formula_column", "minus"]),

("create a calculated column using profit minus tax",
 ["add_formula_column", "minus"]),

("rename column and calculate revenue minus growth",
 ["set_new_column_name", "minus"]),

("open column builder and compute margin minus cost",
 ["open_column_builder", "minus"]),
("upload the file, create a new sheet called variance report, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, add a formula column using revenue minus expenses, rename this column to net revenue, set month as x axis and profit as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "minus",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
# =====================================================
# MULTIPLY (Multiplication Intent)
# =====================================================

("multiply revenue by profit", ["multiply"]),
("revenue multiplied by margin", ["multiply"]),
("calculate sales multiplied by growth", ["multiply"]),
("profit multiplied by tax", ["multiply"]),
("earnings multiplied by margin", ["multiply"]),
("revenue times profit", ["multiply"]),
("sales times growth", ["multiply"]),
("profit times tax", ["multiply"]),
("multiply column a by column b", ["multiply"]),
("compute revenue multiplied by profit", ["multiply"]),

("add a formula column with revenue multiplied by profit",
 ["add_formula_column", "multiply"]),

("create a calculated column using sales multiplied by tax",
 ["add_formula_column", "multiply"]),

("rename column and calculate revenue multiplied by growth",
 ["set_new_column_name", "multiply"]),

("open column builder and compute profit multiplied by margin",
 ["open_column_builder", "multiply"]),

("upload the file, create a new sheet called performance multiplier report, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, add a formula column using revenue multiplied by profit, rename this column to scaled revenue, set month as x axis and profit as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "multiply",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
("product of revenue and profit", ["multiply"]),
("take revenue times profit", ["multiply"]),
("sales times margin", ["multiply"]),
("profit multiplied by revenue", ["multiply"]),
("multiply revenue and profit", ["multiply"]),
("multiply sales and margin", ["multiply"]),
("multiply earnings and growth", ["multiply"]),
("multiply profit and tax", ["multiply"]),
("multiply column revenue by margin", ["multiply"]),
("multiply the revenue with margin", ["multiply"]),
("multiply profit with tax", ["multiply"]),
("multiply growth with revenue", ["multiply"]),
("multiply earnings with margin", ["multiply"]),
("multiply sales with growth", ["multiply"]),
("multiply revenue and profit", ["multiply"]),
("multiply sales and margin", ["multiply"]),
("multiply earnings and growth", ["multiply"]),
("multiply profit and tax", ["multiply"]),
("multiply column revenue by margin", ["multiply"]),
("multiply the revenue with margin", ["multiply"]),
("multiply profit with tax", ["multiply"]),
("multiply growth with revenue", ["multiply"]),
("multiply earnings with margin", ["multiply"]),
("multiply sales with growth", ["multiply"]),
("revenue * profit", ["multiply"]),
("sales * tax", ["multiply"]),
("profit * growth", ["multiply"]),
("margin * revenue", ["multiply"]),
("growth * earnings", ["multiply"]),
("revenue x margin", ["multiply"]),
("sales x tax", ["multiply"]),
("profit x growth", ["multiply"]),
("margin x revenue", ["multiply"]),
("add a formula column using revenue times profit",
 ["add_formula_column", "multiply"]),

("create a calculated column using revenue times margin",
 ["add_formula_column", "multiply"]),

("build a formula column where sales times tax",
 ["add_formula_column", "multiply"]),

("open column builder and multiply revenue by margin",
 ["open_column_builder", "multiply"]),

("open column builder and create a formula column using profit times tax",
 ["open_column_builder", "add_formula_column", "multiply"]),
("upload the spreadsheet, create a sheet called revenue scaling report, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, add a formula column using revenue times margin, rename this column to scaled revenue, set month as x axis and profit as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "multiply",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
("multiply revenue with profit", ["multiply"]),
("multiply sales with margin", ["multiply"]),
("multiply profit with tax", ["multiply"]),
("multiply growth with revenue", ["multiply"]),
("multiply earnings with margin", ["multiply"]),
("multiply revenue against margin", ["multiply"]),
("multiply sales against growth", ["multiply"]),
("multiply profit against tax", ["multiply"]),

("revenue multiplied with profit", ["multiply"]),
("sales multiplied with margin", ["multiply"]),
("profit multiplied with tax", ["multiply"]),
("earnings multiplied with margin", ["multiply"]),

("revenue times margin", ["multiply"]),
("profit times revenue", ["multiply"]),
("sales times tax", ["multiply"]),
("earnings times growth", ["multiply"]),
("margin times revenue", ["multiply"]),

("product of sales and revenue", ["multiply"]),
("product of profit and tax", ["multiply"]),
("product of margin and revenue", ["multiply"]),
("product of earnings and growth", ["multiply"]),

("calculate revenue times margin", ["multiply"]),
("calculate sales times growth", ["multiply"]),
("calculate profit times tax", ["multiply"]),
("calculate earnings times margin", ["multiply"]),

("compute revenue times margin", ["multiply"]),
("compute sales times tax", ["multiply"]),
("compute profit times growth", ["multiply"]),
("compute earnings times revenue", ["multiply"]),

("revenue multiplied against margin", ["multiply"]),
("sales multiplied against growth", ["multiply"]),
("profit multiplied against tax", ["multiply"]),

("scale revenue by margin", ["multiply"]),
("scale sales by growth", ["multiply"]),
("scale profit by tax", ["multiply"]),

("revenue scaled by margin", ["multiply"]),
("sales scaled by growth", ["multiply"]),
("profit scaled by tax", ["multiply"]),

("multiply the revenue and margin values", ["multiply"]),
("multiply the sales and growth columns", ["multiply"]),
("multiply the profit and tax columns", ["multiply"]),

("revenue * margin", ["multiply"]),
("sales * growth", ["multiply"]),
("profit * tax", ["multiply"]),
("earnings * margin", ["multiply"]),

("revenue x profit", ["multiply"]),
("sales x margin", ["multiply"]),
("profit x tax", ["multiply"]),
("growth x revenue", ["multiply"]),

("multiply column revenue with column margin", ["multiply"]),
("multiply column sales with column growth", ["multiply"]),
("multiply column profit with column tax", ["multiply"]),

("take revenue times margin", ["multiply"]),
("take sales times growth", ["multiply"]),
("take profit times tax", ["multiply"]),

("find revenue multiplied by margin", ["multiply"]),
("find sales multiplied by growth", ["multiply"]),
("find profit multiplied by tax", ["multiply"]),

("create a formula column with revenue times margin", ["add_formula_column","multiply"]),

("add a calculated column where sales times growth", ["add_formula_column","multiply"]),

("open column builder and multiply revenue with margin", ["open_column_builder","multiply"]),

("upload file and create a formula column using revenue times margin",
 ["upload_file","add_formula_column","multiply"]),
("create a formula column with revenue times margin", ["add_formula_column","multiply"]),
("add a calculated column where sales times growth", ["add_formula_column","multiply"]),
# =====================================================
# DIVIDE (Division Intent)
# =====================================================

("divide revenue by profit", ["divide"]),
("revenue divided by margin", ["divide"]),
("calculate sales divided by growth", ["divide"]),
("profit divided by tax", ["divide"]),
("earnings divided by margin", ["divide"]),
("revenue divided by cost", ["divide"]),
("sales divided by quantity", ["divide"]),
("profit divided by revenue", ["divide"]),
("divide column a by column b", ["divide"]),
("compute revenue divided by profit", ["divide"]),

# Symbol based
("revenue / profit", ["divide"]),
("sales / tax", ["divide"]),
("profit / growth", ["divide"]),
("margin / revenue", ["divide"]),
("earnings / cost", ["divide"]),

# Natural language variations
("ratio of revenue and profit", ["divide"]),
("ratio of sales and tax", ["divide"]),
("ratio of profit and revenue", ["divide"]),
("revenue per profit", ["divide"]),
("profit per revenue", ["divide"]),
("sales per unit", ["divide"]),
("earnings per cost", ["divide"]),
("margin per revenue", ["divide"]),

# Formula column combinations
("add a formula column with revenue divided by profit",
 ["add_formula_column", "divide"]),

("create a calculated column using sales divided by tax",
 ["add_formula_column", "divide"]),

("build a formula column where profit divided by revenue",
 ["add_formula_column", "divide"]),

("rename column and calculate revenue divided by growth",
 ["set_new_column_name", "divide"]),

("open column builder and compute profit divided by margin",
 ["open_column_builder", "divide"]),

# Triple combinations
("open column builder and create a formula column using revenue divided by margin",
 ["open_column_builder", "add_formula_column", "divide"]),

("upload file and create a formula column using revenue divided by margin",
 ["upload_file", "add_formula_column", "divide"]),

# Full workflow example
("upload the spreadsheet, create a sheet called ratio analysis report, set it as base sheet, define cleaned data as preprocessing sheet and final output as postprocessing sheet, add a formula column using revenue divided by profit, rename this column to profit ratio, set month as x axis and margin as y axis, and open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "add_formula_column",
  "divide",
  "set_new_column_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
# =====================================================
# MORE DIVIDE VARIATIONS
# =====================================================

("divide revenue with profit", ["divide"]),
("divide sales with tax", ["divide"]),
("divide profit with revenue", ["divide"]),
("divide margin with revenue", ["divide"]),

("revenue divided with profit", ["divide"]),
("sales divided with tax", ["divide"]),
("profit divided with revenue", ["divide"]),

("calculate revenue per unit", ["divide"]),
("calculate profit per sale", ["divide"]),
("calculate earnings per cost", ["divide"]),

("compute revenue per employee", ["divide"]),
("compute profit per order", ["divide"]),
("compute sales per region", ["divide"]),

("find revenue per profit", ["divide"]),
("find profit per revenue", ["divide"]),
("find sales per quantity", ["divide"]),

("revenue over profit", ["divide"]),
("sales over tax", ["divide"]),
("profit over revenue", ["divide"]),
("earnings over cost", ["divide"]),

("revenue compared to profit", ["divide"]),
("sales compared to tax", ["divide"]),
("profit compared to revenue", ["divide"]),

("take revenue divided by margin", ["divide"]),
("take sales divided by tax", ["divide"]),
("take profit divided by revenue", ["divide"]),

("determine revenue divided by cost", ["divide"]),
("determine profit divided by revenue", ["divide"]),

("revenue divided against profit", ["divide"]),
("sales divided against tax", ["divide"]),
("profit divided against revenue", ["divide"]),

("divide the revenue by the profit", ["divide"]),
("divide the sales by the tax", ["divide"]),
("divide the profit by the revenue", ["divide"]),

("calculate the ratio between revenue and profit", ["divide"]),
("calculate the ratio between sales and tax", ["divide"]),
("calculate the ratio between profit and revenue", ["divide"]),
("create a formula column using revenue divided by margin",
 ["add_formula_column","divide"]),

("add a calculated column where profit divided by revenue",
 ["add_formula_column","divide"]),

("open column builder and divide revenue by margin",
 ["open_column_builder","divide"]),

("open column builder and create a formula column using profit divided by revenue",
 ["open_column_builder","add_formula_column","divide"]),
# =====================================================
# ADDITIONAL DIVIDE VARIATIONS
# =====================================================

("divide revenue by margin", ["divide"]),
("divide sales by profit", ["divide"]),
("divide earnings by revenue", ["divide"]),
("divide margin by revenue", ["divide"]),
("divide profit by cost", ["divide"]),

("revenue divided by sales", ["divide"]),
("sales divided by revenue", ["divide"]),
("profit divided by margin", ["divide"]),
("margin divided by earnings", ["divide"]),
("earnings divided by profit", ["divide"]),

("calculate revenue per customer", ["divide"]),
("calculate profit per transaction", ["divide"]),
("calculate revenue per region", ["divide"]),
("calculate earnings per department", ["divide"]),
("calculate sales per order", ["divide"]),

("compute revenue per employee", ["divide"]),
("compute sales per store", ["divide"]),
("compute profit per unit", ["divide"]),
("compute margin per revenue", ["divide"]),
("compute earnings per order", ["divide"]),

("find revenue per sale", ["divide"]),
("find profit per customer", ["divide"]),
("find earnings per department", ["divide"]),
("find margin per revenue", ["divide"]),
("find sales per region", ["divide"]),

("revenue over sales", ["divide"]),
("profit over cost", ["divide"]),
("earnings over revenue", ["divide"]),
("margin over profit", ["divide"]),
("sales over quantity", ["divide"]),

("revenue relative to profit", ["divide"]),
("profit relative to revenue", ["divide"]),
("sales relative to growth", ["divide"]),
("earnings relative to cost", ["divide"]),
("margin relative to revenue", ["divide"]),

("normalize revenue by profit", ["divide"]),
("normalize sales by growth", ["divide"]),
("normalize profit by revenue", ["divide"]),
("normalize earnings by cost", ["divide"]),
("normalize margin by revenue", ["divide"]),

("revenue ratio to profit", ["divide"]),
("profit ratio to revenue", ["divide"]),
("sales ratio to tax", ["divide"]),
("earnings ratio to cost", ["divide"]),
("margin ratio to revenue", ["divide"]),

("determine revenue per order", ["divide"]),
("determine profit per customer", ["divide"]),
("determine earnings per employee", ["divide"]),
("determine margin per revenue", ["divide"]),
("determine sales per region", ["divide"]),

("revenue divided across profit", ["divide"]),
("sales divided across tax", ["divide"]),
("profit divided across revenue", ["divide"]),
("earnings divided across cost", ["divide"]),
("margin divided across revenue", ["divide"]),

("break revenue by profit ratio", ["divide"]),
("break profit by revenue ratio", ["divide"]),
("break sales by tax ratio", ["divide"]),
("break earnings by cost ratio", ["divide"]),
("break margin by revenue ratio", ["divide"]),

("revenue / margin", ["divide"]),
("profit / cost", ["divide"]),
("sales / revenue", ["divide"]),
("earnings / profit", ["divide"]),
("margin / earnings", ["divide"]),

("create a ratio column using revenue divided by profit", ["add_formula_column","divide"]),
("build a formula column where sales divided by tax", ["add_formula_column","divide"]),
("add a computed column using profit divided by revenue", ["add_formula_column","divide"]),
("insert a calculated column with earnings divided by cost", ["add_formula_column","divide"]),
("generate a formula column using margin divided by revenue", ["add_formula_column","divide"]),

("open column builder and divide revenue by profit", ["open_column_builder","divide"]),
("open column builder and compute sales divided by tax", ["open_column_builder","divide"]),
("open column builder and create a ratio using profit divided by revenue", ["open_column_builder","divide"]),
("open column builder and calculate earnings divided by cost", ["open_column_builder","divide"]),

("upload file and compute revenue divided by profit", ["upload_file","divide"]),
("upload dataset and calculate profit divided by revenue", ["upload_file","divide"]),
("import spreadsheet and determine sales divided by tax", ["upload_file","divide"]),
("load dataset and find earnings divided by cost", ["upload_file","divide"]),

("upload file and create a formula column with revenue divided by profit",
 ["upload_file","add_formula_column","divide"]),

("create sheet and compute sales divided by tax",
 ["create_new_sheet","divide"]),

("rename column and compute profit divided by revenue",
 ["set_new_column_name","divide"]),
# =====================================================
# EXTRA DIVIDE INTENT TRAINING DATA
# =====================================================

# --- Per / Average style ---
("revenue per employee", ["divide"]),
("profit per employee", ["divide"]),
("sales per customer", ["divide"]),
("revenue per order", ["divide"]),
("profit per order", ["divide"]),
("sales per transaction", ["divide"]),
("earnings per share", ["divide"]),
("revenue per unit", ["divide"]),
("profit per product", ["divide"]),
("sales per region", ["divide"]),
("cost per unit", ["divide"]),
("cost per customer", ["divide"]),
("revenue per store", ["divide"]),
("profit per store", ["divide"]),
("sales per store", ["divide"]),

# --- Average phrasing ---
("average revenue per customer", ["divide"]),
("average profit per order", ["divide"]),
("average sales per store", ["divide"]),
("average earnings per employee", ["divide"]),
("average revenue per transaction", ["divide"]),

# --- Ratio phrasing ---
("revenue to profit ratio", ["divide"]),
("profit to revenue ratio", ["divide"]),
("sales to cost ratio", ["divide"]),
("revenue to cost ratio", ["divide"]),
("profit to expense ratio", ["divide"]),
("margin to revenue ratio", ["divide"]),
("tax to revenue ratio", ["divide"]),
("cost to revenue ratio", ["divide"]),
("expense to revenue ratio", ["divide"]),

# --- Ratio calculation ---
("calculate revenue to profit ratio", ["divide"]),
("calculate profit to revenue ratio", ["divide"]),
("calculate sales to cost ratio", ["divide"]),
("calculate revenue to cost ratio", ["divide"]),
("compute revenue to profit ratio", ["divide"]),
("compute profit to revenue ratio", ["divide"]),

# --- Rate phrasing ---
("revenue rate per employee", ["divide"]),
("profit rate per order", ["divide"]),
("sales rate per region", ["divide"]),
("growth rate per quarter", ["divide"]),

# --- Percent style ---
("profit as percentage of revenue", ["divide"]),
("cost as percentage of revenue", ["divide"]),
("tax as percentage of profit", ["divide"]),
("margin as percentage of revenue", ["divide"]),
("expense as percentage of revenue", ["divide"]),

# --- Compare phrasing ---
("compare revenue to profit", ["divide"]),
("compare sales to revenue", ["divide"]),
("compare profit to cost", ["divide"]),
("compare revenue with expenses", ["divide"]),

# --- Fraction style ---
("fraction of revenue to profit", ["divide"]),
("fraction of profit to revenue", ["divide"]),
("fraction of sales to tax", ["divide"]),

# --- Efficiency phrasing ---
("revenue efficiency against cost", ["divide"]),
("profit efficiency against revenue", ["divide"]),
("sales efficiency against tax", ["divide"]),

# --- Scaling / normalization ---
("normalize revenue by cost", ["divide"]),
("normalize profit by revenue", ["divide"]),
("normalize sales by growth", ["divide"]),
("normalize earnings by expenses", ["divide"]),

# --- Dataset phrasing ---
("revenue divided by number of employees", ["divide"]),
("profit divided by total revenue", ["divide"]),
("sales divided by number of customers", ["divide"]),
("earnings divided by total cost", ["divide"]),

# --- Symbol variants ---
("revenue ÷ profit", ["divide"]),
("sales ÷ tax", ["divide"]),
("profit ÷ revenue", ["divide"]),
("earnings ÷ cost", ["divide"]),
("margin ÷ revenue", ["divide"]),

# --- Column phrasing ---
("divide the revenue column by profit column", ["divide"]),
("divide the sales column by tax column", ["divide"]),
("divide the profit column by revenue column", ["divide"]),
("divide the earnings column by cost column", ["divide"]),
("revenue/profit", ["divide"]),
("profit/revenue", ["divide"]),
("sales/tax", ["divide"]),
("revenue divided profit", ["divide"]),
("profit divided revenue", ["divide"]),
("revenue/profit", ["divide"]),
("profit/revenue", ["divide"]),
("sales/tax", ["divide"]),
("revenue divided profit", ["divide"]),
("profit divided revenue", ["divide"]),
# =====================================================
# MORE DIVIDE TRAINING DATA
# =====================================================

# --- Performance ratios ---
("revenue divided by employees", ["divide"]),
("profit divided by employees", ["divide"]),
("sales divided by employees", ["divide"]),
("earnings divided by employees", ["divide"]),
("revenue divided by number of customers", ["divide"]),
("profit divided by number of orders", ["divide"]),

# --- Productivity metrics ---
("revenue per employee ratio", ["divide"]),
("profit per employee ratio", ["divide"]),
("sales per employee ratio", ["divide"]),
("earnings per employee ratio", ["divide"]),

# --- Operational metrics ---
("revenue per day", ["divide"]),
("revenue per month", ["divide"]),
("revenue per quarter", ["divide"]),
("profit per month", ["divide"]),
("sales per week", ["divide"]),

# --- Efficiency metrics ---
("revenue efficiency per employee", ["divide"]),
("profit efficiency per order", ["divide"]),
("sales efficiency per region", ["divide"]),
("earnings efficiency per department", ["divide"]),

# --- Margin style phrasing ---
("revenue compared to expenses", ["divide"]),
("profit compared to cost", ["divide"]),
("sales compared to revenue", ["divide"]),
("earnings compared to margin", ["divide"]),

# --- Analytical phrasing ---
("analyze revenue over expenses", ["divide"]),
("analyze profit over cost", ["divide"]),
("analyze sales over tax", ["divide"]),
("analyze earnings over revenue", ["divide"]),

# --- Benchmark style ---
("benchmark revenue against cost", ["divide"]),
("benchmark profit against revenue", ["divide"]),
("benchmark sales against tax", ["divide"]),
("benchmark earnings against expenses", ["divide"]),

# --- Scaling ratios ---
("scale revenue relative to profit", ["divide"]),
("scale sales relative to tax", ["divide"]),
("scale profit relative to revenue", ["divide"]),
("scale earnings relative to cost", ["divide"]),

# --- Data science style ---
("normalize revenue against profit", ["divide"]),
("normalize sales against tax", ["divide"]),
("normalize profit against revenue", ["divide"]),
("normalize earnings against cost", ["divide"]),

# --- Relative comparison ---
("revenue relative to cost", ["divide"]),
("profit relative to tax", ["divide"]),
("sales relative to revenue", ["divide"]),
("earnings relative to margin", ["divide"]),

# --- Ratio building ---
("build revenue to profit ratio", ["divide"]),
("build profit to revenue ratio", ["divide"]),
("build sales to cost ratio", ["divide"]),
("build earnings to margin ratio", ["divide"]),

# --- KPI language ---
("revenue kpi per employee", ["divide"]),
("profit kpi per order", ["divide"]),
("sales kpi per store", ["divide"]),
("earnings kpi per region", ["divide"]),

# --- Reporting language ---
("report revenue per customer", ["divide"]),
("report profit per transaction", ["divide"]),
("report sales per region", ["divide"]),
("report earnings per department", ["divide"]),

# --- Financial analysis ---
("financial ratio revenue to cost", ["divide"]),
("financial ratio profit to revenue", ["divide"]),
("financial ratio sales to tax", ["divide"]),
("financial ratio earnings to margin", ["divide"]),

# --- Evaluation language ---
("evaluate revenue against expenses", ["divide"]),
("evaluate profit against revenue", ["divide"]),
("evaluate sales against tax", ["divide"]),
("evaluate earnings against cost", ["divide"]),

# --- Column focused ---
("divide revenue column by profit column", ["divide"]),
("divide sales column by tax column", ["divide"]),
("divide profit column by revenue column", ["divide"]),
("divide earnings column by cost column", ["divide"]),

# --- Instruction style ---
("take revenue divided by cost", ["divide"]),
("take profit divided by revenue", ["divide"]),
("take sales divided by tax", ["divide"]),
("take earnings divided by margin", ["divide"]),

# --- Data analysis phrasing ---
("measure revenue per customer", ["divide"]),
("measure profit per transaction", ["divide"]),
("measure sales per region", ["divide"]),
("measure earnings per department", ["divide"]),

# --- Performance indicators ---
("revenue per staff member", ["divide"]),
("profit per staff member", ["divide"]),
("sales per staff member", ["divide"]),
("earnings per staff member", ["divide"]),
# =====================================================
# FULL SENTENCE DIVIDE INTENTS
# =====================================================

# --- per phrasing ---
("calculate revenue per employee", ["divide"]),
("calculate profit per employee", ["divide"]),
("calculate sales per employee", ["divide"]),
("calculate revenue per customer", ["divide"]),
("calculate profit per order", ["divide"]),
("calculate sales per transaction", ["divide"]),
("calculate earnings per store", ["divide"]),
("calculate revenue per region", ["divide"]),

("show me revenue per employee", ["divide"]),
("show me profit per customer", ["divide"]),
("show me sales per store", ["divide"]),
("show me earnings per department", ["divide"]),

("compute revenue per employee in the dataset", ["divide"]),
("compute profit per order for this sheet", ["divide"]),
("compute sales per store from this data", ["divide"]),
("compute earnings per department from the table", ["divide"]),

# --- over phrasing ---
("calculate revenue over expenses", ["divide"]),
("calculate profit over revenue", ["divide"]),
("calculate sales over cost", ["divide"]),
("calculate earnings over tax", ["divide"]),

("show revenue over expenses in the dataset", ["divide"]),
("show profit over revenue in this sheet", ["divide"]),
("show sales over cost for this table", ["divide"]),
("show earnings over tax from the data", ["divide"]),

# --- ratio phrasing ---
("calculate the ratio of revenue to cost", ["divide"]),
("calculate the ratio of profit to revenue", ["divide"]),
("calculate the ratio of sales to expenses", ["divide"]),
("calculate the ratio of earnings to margin", ["divide"]),

("show the revenue to cost ratio", ["divide"]),
("show the profit to revenue ratio", ["divide"]),
("show the sales to expense ratio", ["divide"]),
("show the earnings to margin ratio", ["divide"]),

("find the revenue to cost ratio in the dataset", ["divide"]),
("find the profit to revenue ratio in this sheet", ["divide"]),
("find the sales to tax ratio in the table", ["divide"]),
("find the earnings to margin ratio in the data", ["divide"]),

# --- out of phrasing ---
("calculate profit out of revenue", ["divide"]),
("calculate tax out of sales", ["divide"]),
("calculate expenses out of revenue", ["divide"]),
("calculate margin out of earnings", ["divide"]),

("show profit out of revenue for this data", ["divide"]),
("show tax out of sales in the dataset", ["divide"]),
("show expenses out of revenue in the sheet", ["divide"]),
("show margin out of earnings in the table", ["divide"]),

# --- compared phrasing ---
("compare revenue against cost", ["divide"]),
("compare profit against revenue", ["divide"]),
("compare sales against tax", ["divide"]),
("compare earnings against margin", ["divide"]),

("calculate revenue compared with cost", ["divide"]),
("calculate profit compared with revenue", ["divide"]),
("calculate sales compared with tax", ["divide"]),
("calculate earnings compared with margin", ["divide"]),

# --- relative phrasing ---
("calculate revenue relative to cost", ["divide"]),
("calculate profit relative to revenue", ["divide"]),
("calculate sales relative to tax", ["divide"]),
("calculate earnings relative to margin", ["divide"]),

("show revenue relative to cost in this dataset", ["divide"]),
("show profit relative to revenue in the sheet", ["divide"]),
("show sales relative to tax in the table", ["divide"]),
("show earnings relative to margin in the data", ["divide"]),

# --- for every phrasing ---
("calculate revenue for every employee", ["divide"]),
("calculate profit for every order", ["divide"]),
("calculate sales for every store", ["divide"]),
("calculate earnings for every department", ["divide"]),

("show revenue for every employee in the dataset", ["divide"]),
("show profit for every order in the sheet", ["divide"]),
("show sales for every store in the table", ["divide"]),
("show earnings for every department in the data", ["divide"]),

# --- instruction style ---
("take revenue divided by cost", ["divide"]),
("take profit divided by revenue", ["divide"]),
("take sales divided by tax", ["divide"]),
("take earnings divided by margin", ["divide"]),

("create a value that divides revenue by cost", ["divide"]),
("create a value that divides profit by revenue", ["divide"]),
("create a value that divides sales by tax", ["divide"]),
("create a value that divides earnings by margin", ["divide"]),

# --- analytical phrasing ---
("analyze revenue per employee across the dataset", ["divide"]),
("analyze profit per order in this sheet", ["divide"]),
("analyze sales per store in the table", ["divide"]),
("analyze earnings per department in the data", ["divide"]),

("evaluate revenue relative to expenses", ["divide"]),
("evaluate profit relative to revenue", ["divide"]),
("evaluate sales relative to tax", ["divide"]),
("evaluate earnings relative to margin", ["divide"]),

# --- column language ---
("divide the revenue column by the cost column", ["divide"]),
("divide the profit column by the revenue column", ["divide"]),
("divide the sales column by the tax column", ["divide"]),
("divide the earnings column by the margin column", ["divide"]),

("calculate revenue per employee using the revenue and employee columns", ["divide"]),
("calculate profit per order using the profit and order columns", ["divide"]),
("calculate sales per store using the sales and store columns", ["divide"]),
("calculate earnings per department using the earnings and department columns", ["divide"]),
# =====================================================
# LEFT BRACKET (Opening Parenthesis Intent)
# =====================================================

("open bracket", ["left_bracket"]),
("add a left bracket", ["left_bracket"]),
("insert a left bracket", ["left_bracket"]),
("start a bracket", ["left_bracket"]),
("start with a bracket", ["left_bracket"]),
("begin a bracket", ["left_bracket"]),
("use an opening bracket", ["left_bracket"]),
("place a left parenthesis", ["left_bracket"]),
("insert opening parenthesis", ["left_bracket"]),
("add opening bracket", ["left_bracket"]),

# Symbol usage
("(", ["left_bracket"]),
("start formula with (", ["left_bracket"]),
("add ( before revenue", ["left_bracket"]),
("put ( before profit", ["left_bracket"]),
("insert ( in formula", ["left_bracket"]),

# Formula building language
("open bracket before revenue plus profit", ["left_bracket"]),
("start bracket for revenue plus profit", ["left_bracket"]),
("begin bracket around revenue and profit", ["left_bracket"]),
("add bracket around revenue plus margin", ["left_bracket"]),
("group revenue and profit with bracket", ["left_bracket"]),

# Natural phrasing
("start grouping revenue and profit", ["left_bracket"]),
("group revenue and margin together", ["left_bracket"]),
("begin grouping sales and tax", ["left_bracket"]),
("group profit and cost first", ["left_bracket"]),

# Column builder combinations
("open column builder and add a left bracket",
 ["open_column_builder","left_bracket"]),

("open column builder and insert opening bracket",
 ["open_column_builder","left_bracket"]),

# Formula column combinations
("add a formula column starting with a bracket",
 ["add_formula_column","left_bracket"]),

("create a calculated column with an opening bracket",
 ["add_formula_column","left_bracket"]),

("build a formula column starting with (",
 ["add_formula_column","left_bracket"]),

# Mixed arithmetic scenarios
("open bracket then add revenue plus profit",
 ["left_bracket","plus"]),

("open bracket and subtract cost from revenue",
 ["left_bracket","minus"]),

("open bracket and multiply revenue by margin",
 ["left_bracket","multiply"]),

("open bracket and divide revenue by profit",
 ["left_bracket","divide"]),

# Natural spoken instructions
("put revenue and profit inside brackets",
 ["left_bracket"]),

("wrap revenue and margin in brackets",
 ["left_bracket"]),

("start the expression with a bracket",
 ["left_bracket"]),

("add parentheses before calculating revenue plus profit",
 ["left_bracket"]),

# Workflow example
("upload file and create a formula column starting with a bracket",
 ["upload_file","add_formula_column","left_bracket"]),

("open column builder and start the formula with (",
 ["open_column_builder","left_bracket"]),

("create a calculated column and begin with a bracket",
 ["create_new_sheet","add_formula_column","left_bracket"]),
# =====================================================
# RIGHT CLICK INTENT
# =====================================================

# =====================================================
# RIGHT BRACKET INTENT
# =====================================================

("close the bracket", ["right_bracket"]),
("add a closing bracket", ["right_bracket"]),
("insert a right bracket", ["right_bracket"]),
("place a right bracket", ["right_bracket"]),
("add a closing parenthesis", ["right_bracket"]),
("insert a closing parenthesis", ["right_bracket"]),
("put a right parenthesis", ["right_bracket"]),
("close the parentheses", ["right_bracket"]),
("finish the bracket", ["right_bracket"]),
("end the bracket", ["right_bracket"]),
("complete the bracket", ["right_bracket"]),
("add the closing bracket now", ["right_bracket"]),
("insert the final bracket", ["right_bracket"]),
("add the right parenthesis here", ["right_bracket"]),
("close this bracket", ["right_bracket"]),
("close the expression with a bracket", ["right_bracket"]),
("finish the expression with a right bracket", ["right_bracket"]),
("complete the formula with a closing bracket", ["right_bracket"]),
("end the formula with a bracket", ["right_bracket"]),
("add the closing parenthesis to the formula", ["right_bracket"]),
("close the math expression", ["right_bracket"]),
("finish the parenthesis", ["right_bracket"]),
("place the closing parenthesis", ["right_bracket"]),
("insert the closing bracket here", ["right_bracket"]),
("put the right bracket at the end", ["right_bracket"]),
("close the calculation bracket", ["right_bracket"]),
("finish the grouped calculation with a bracket", ["right_bracket"]),
("add the bracket to close the group", ["right_bracket"]),
("end the grouped expression", ["right_bracket"]),
("close the group with a parenthesis", ["right_bracket"]),
("complete the grouping with a bracket", ["right_bracket"]),
("wrap up the expression with a right bracket", ["right_bracket"]),
("close the parentheses now", ["right_bracket"]),
("finish the bracketed expression", ["right_bracket"]),
("terminate the bracket", ["right_bracket"]),
("close this parenthesis", ["right_bracket"]),
("add the ending bracket", ["right_bracket"]),
("insert the final parenthesis", ["right_bracket"]),
("place the ending bracket", ["right_bracket"]),
("close the current bracket", ["right_bracket"]),
# =====================================================
# CLEAR COLUMN FORMULA INTENT
# =====================================================

("clear the column formula", ["clear_column_formula"]),
("remove the formula from the column", ["clear_column_formula"]),
("delete the column formula", ["clear_column_formula"]),
("reset the column formula", ["clear_column_formula"]),
("erase the column formula", ["clear_column_formula"]),
("remove formula from this column", ["clear_column_formula"]),
("clear formula from the selected column", ["clear_column_formula"]),
("delete the formula in this column", ["clear_column_formula"]),
("reset the formula in the column", ["clear_column_formula"]),
("clear the current formula", ["clear_column_formula"]),

# Column builder phrasing
("clear the formula in column builder", ["clear_column_formula"]),
("remove the formula from column builder", ["clear_column_formula"]),
("reset the formula in column builder", ["clear_column_formula"]),
("delete the formula from the column builder", ["clear_column_formula"]),
("erase the formula inside column builder", ["clear_column_formula"]),

# Natural instructions
("remove the calculation from this column", ["clear_column_formula"]),
("clear the calculation in the column", ["clear_column_formula"]),
("delete the calculation from the column", ["clear_column_formula"]),
("remove the expression from the column", ["clear_column_formula"]),
("clear the expression from this column", ["clear_column_formula"]),
("delete the expression from the column", ["clear_column_formula"]),

# Formula editing language
("clear the formula I wrote", ["clear_column_formula"]),
("remove the formula I just added", ["clear_column_formula"]),
("delete the current formula", ["clear_column_formula"]),
("erase the current formula", ["clear_column_formula"]),
("reset the current expression", ["clear_column_formula"]),

# UI style phrasing
("clear formula", ["clear_column_formula"]),
("remove formula", ["clear_column_formula"]),
("delete formula", ["clear_column_formula"]),
("reset formula", ["clear_column_formula"]),
("erase formula", ["clear_column_formula"]),

# Workflow combinations
("open column builder and clear the formula",
 ["open_column_builder","clear_column_formula"]),

("clear the column formula and start again",
 ["clear_column_formula"]),

("clear the formula and add a new one",
 ["clear_column_formula"]),

("upload file and clear the column formula",
 ["upload_file","clear_column_formula"]),

("open column builder and remove the formula from the column",
 ["open_column_builder","clear_column_formula"]),

# Full sentence instructions
("please clear the formula from the column", ["clear_column_formula"]),
("can you remove the formula from this column", ["clear_column_formula"]),
("i want to delete the formula in this column", ["clear_column_formula"]),
("remove the existing formula so I can write a new one", ["clear_column_formula"]),
("reset the column formula so it becomes empty", ["clear_column_formula"]),
("clear the formula but keep the column", ["clear_column_formula"]),
("remove the formula not the column", ["clear_column_formula"]),
# =====================================================
# REMOVE LAST VARIABLE INTENT
# =====================================================

("remove the last variable", ["remove_last_variable"]),
("delete the last variable", ["remove_last_variable"]),
("remove the previous variable", ["remove_last_variable"]),
("remove the last column variable", ["remove_last_variable"]),
("delete the last column variable", ["remove_last_variable"]),
("remove the last field from the formula", ["remove_last_variable"]),
("delete the last field from the formula", ["remove_last_variable"]),
("remove the last item in the formula", ["remove_last_variable"]),
("remove the last element from the formula", ["remove_last_variable"]),
("remove the last entry in the formula", ["remove_last_variable"]),

# Builder style phrasing
("remove the last variable from the column builder", ["remove_last_variable"]),
("delete the last variable from column builder", ["remove_last_variable"]),
("remove the last variable in the expression", ["remove_last_variable"]),
("delete the last variable in the expression", ["remove_last_variable"]),
("remove the last part of the formula", ["remove_last_variable"]),
("delete the last part of the formula", ["remove_last_variable"]),

# Natural phrasing
("take out the last variable", ["remove_last_variable"]),
("take out the last column from the formula", ["remove_last_variable"]),
("remove the last thing I added", ["remove_last_variable"]),
("delete the last thing I added to the formula", ["remove_last_variable"]),
("remove the most recent variable", ["remove_last_variable"]),
("delete the most recent variable", ["remove_last_variable"]),

# Undo style
("undo the last variable", ["remove_last_variable"]),
("undo the last variable addition", ["remove_last_variable"]),
("remove the variable I just added", ["remove_last_variable"]),
("delete the variable I just inserted", ["remove_last_variable"]),
("remove the column I just added to the formula", ["remove_last_variable"]),
("delete the column I just added to the expression", ["remove_last_variable"]),

# Formula editing
("remove the last variable from this calculation", ["remove_last_variable"]),
("delete the last variable from this calculation", ["remove_last_variable"]),
("remove the last variable from the expression", ["remove_last_variable"]),
("delete the last variable from the expression", ["remove_last_variable"]),
("remove the last operand", ["remove_last_variable"]),
("delete the last operand", ["remove_last_variable"]),

# Arithmetic contexts
("remove the last variable from revenue plus profit", ["remove_last_variable"]),
("delete the last variable from revenue minus cost", ["remove_last_variable"]),
("remove the last variable from sales multiplied by margin", ["remove_last_variable"]),
("remove the last variable from revenue divided by tax", ["remove_last_variable"]),

# Column builder combinations
("open column builder and remove the last variable",
 ["open_column_builder","remove_last_variable"]),

("open column builder and delete the last variable",
 ["open_column_builder","remove_last_variable"]),

# Formula column combinations
("add a formula column and remove the last variable",
 ["add_formula_column","remove_last_variable"]),

("create a calculated column then remove the last variable",
 ["add_formula_column","remove_last_variable"]),

# Workflow examples
("upload file and remove the last variable from the formula",
 ["upload_file","remove_last_variable"]),

("open column builder and remove the variable I just added",
 ["open_column_builder","remove_last_variable"]),

("create a formula column and delete the last variable",
 ["add_formula_column","remove_last_variable"]),

# Short UI commands
("remove last variable", ["remove_last_variable"]),
("delete last variable", ["remove_last_variable"]),
("undo last variable", ["remove_last_variable"]),
("remove last column", ["remove_last_variable"]),
("delete last column from formula", ["remove_last_variable"]),
("remove only the last variable not the whole formula", ["remove_last_variable"]),
("delete the last variable but keep the formula", ["remove_last_variable"]),


("submit the column", ["submit_column"]),
("submit this column", ["submit_column"]),
("submit the formula column", ["submit_column"]),
("submit the new column", ["submit_column"]),
("submit the calculated column", ["submit_column"]),

("save the column", ["submit_column"]),
("save this column", ["submit_column"]),
("save the new column", ["submit_column"]),
("save the formula column", ["submit_column"]),
("save the calculated column", ["submit_column"]),

("apply the column", ["submit_column"]),
("apply this column", ["submit_column"]),
("apply the formula column", ["submit_column"]),
("apply the new column", ["submit_column"]),

("finish creating the column", ["submit_column"]),
("finish the column", ["submit_column"]),
("complete the column", ["submit_column"]),
("complete the formula column", ["submit_column"]),
("complete column creation", ["submit_column"]),

("confirm the column", ["submit_column"]),
("confirm this column", ["submit_column"]),
("confirm the formula column", ["submit_column"]),

("add the column", ["submit_column"]),
("add this column to the sheet", ["submit_column"]),
("add the formula column to the sheet", ["submit_column"]),
("insert the column", ["submit_column"]),

("finalize the column", ["submit_column"]),
("finalize the formula column", ["submit_column"]),
("finalize this column", ["submit_column"]),

("done creating the column", ["submit_column"]),
("done with the column", ["submit_column"]),
("done with this column", ["submit_column"]),

("submit column now", ["submit_column"]),
("save and submit the column", ["submit_column"]),
("submit the column to the sheet", ["submit_column"]),
("submit the column builder result", ["submit_column"]),

("create the column and submit it", ["add_formula_column","submit_column"]),
("open column builder and submit the column", ["open_column_builder","submit_column"]),
("finish the formula and submit the column", ["submit_column"]),


("submit the sheet", ["submit_sheet"]),
("submit this sheet", ["submit_sheet"]),
("submit the new sheet", ["submit_sheet"]),
("submit my sheet", ["submit_sheet"]),

("save the sheet", ["submit_sheet"]),
("save this sheet", ["submit_sheet"]),
("save the new sheet", ["submit_sheet"]),
("save my sheet", ["submit_sheet"]),

("apply the sheet", ["submit_sheet"]),
("apply this sheet", ["submit_sheet"]),
("apply the new sheet", ["submit_sheet"]),

("finish creating the sheet", ["submit_sheet"]),
("finish the sheet", ["submit_sheet"]),
("complete the sheet", ["submit_sheet"]),
("complete sheet creation", ["submit_sheet"]),

("confirm the sheet", ["submit_sheet"]),
("confirm this sheet", ["submit_sheet"]),
("confirm the new sheet", ["submit_sheet"]),

("finalize the sheet", ["submit_sheet"]),
("finalize this sheet", ["submit_sheet"]),
("finalize the new sheet", ["submit_sheet"]),

("generate the sheet", ["submit_sheet"]),
("generate this sheet", ["submit_sheet"]),
("generate the final sheet", ["submit_sheet"]),

("done creating the sheet", ["submit_sheet"]),
("done with the sheet", ["submit_sheet"]),
("done with this sheet", ["submit_sheet"]),

("submit sheet now", ["submit_sheet"]),
("submit the sheet to dashboard", ["submit_sheet"]),
("submit the sheet to the workspace", ["submit_sheet"]),

("create the sheet and submit it", ["create_new_sheet","submit_sheet"]),
("finish configuring the sheet and submit it", ["submit_sheet"]),
("complete the dashboard and submit the sheet", ["submit_sheet"]),

("submit the sheet after setting axes", ["set_x_axis","set_y_axis","submit_sheet"]),
("submit the sheet after creating columns", ["submit_column","submit_sheet"]),
("submit the sheet once everything is ready", ["submit_sheet"]),
("submit the sheet not the column", ["submit_sheet"]),
("save the whole sheet", ["submit_sheet"]),
("finish and submit the entire sheet", ["submit_sheet"]),
("submit the completed sheet", ["submit_sheet"]),
("submit the final sheet", ["submit_sheet"]),

("go to results", ["go_to_results"]),
("open results", ["go_to_results"]),
("show results", ["go_to_results"]),
("view results", ["go_to_results"]),
("take me to the results", ["go_to_results"]),

("navigate to results", ["go_to_results"]),
("navigate to the results page", ["go_to_results"]),
("go to the results page", ["go_to_results"]),
("open the results page", ["go_to_results"]),
("show the results page", ["go_to_results"]),

("view the results", ["go_to_results"]),
("display the results", ["go_to_results"]),
("show me the results", ["go_to_results"]),
("let me see the results", ["go_to_results"]),
("can I see the results", ["go_to_results"]),

("go to output", ["go_to_results"]),
("open output page", ["go_to_results"]),
("show the output", ["go_to_results"]),
("view the output", ["go_to_results"]),

("open the results dashboard", ["go_to_results"]),
("go to the results dashboard", ["go_to_results"]),
("show the results dashboard", ["go_to_results"]),
("view the results dashboard", ["go_to_results"]),

("take me to the dashboard results", ["go_to_results"]),
("navigate to dashboard results", ["go_to_results"]),

("after submitting the sheet go to results", ["submit_sheet","go_to_results"]),
("submit sheet and go to results", ["submit_sheet","go_to_results"]),
("once finished open results", ["go_to_results"]),
("finish and show results", ["go_to_results"]),

("show me the final results", ["go_to_results"]),
("open the final results page", ["go_to_results"]),
("view the final results", ["go_to_results"]),
("display final results", ["go_to_results"]),
("the sheet is done now go to results", ["go_to_results"]),
("stop editing and show results", ["go_to_results"]),
("switch to results view", ["go_to_results"]),
("move to results page", ["go_to_results"]),
# =====================================================
# MORE PLUS INTENT DATA
# =====================================================

("revenue plus profit", ["plus"]),
("sales plus tax", ["plus"]),
("profit plus margin", ["plus"]),
("earnings plus bonus", ["plus"]),
("income plus interest", ["plus"]),
("growth plus revenue", ["plus"]),
("cost plus tax", ["plus"]),
("revenue plus expenses", ["plus"]),
("sales plus revenue", ["plus"]),
("profit plus earnings", ["plus"]),

("add revenue plus profit", ["plus"]),
("add sales plus tax", ["plus"]),
("add profit plus margin", ["plus"]),
("add earnings plus bonus", ["plus"]),
("add income plus interest", ["plus"]),
("add growth plus revenue", ["plus"]),
("add cost plus tax", ["plus"]),
("add revenue plus expenses", ["plus"]),
("add sales plus revenue", ["plus"]),
("add profit plus earnings", ["plus"]),

("calculate revenue plus profit", ["plus"]),
("calculate sales plus tax", ["plus"]),
("calculate profit plus margin", ["plus"]),
("calculate earnings plus bonus", ["plus"]),
("calculate income plus interest", ["plus"]),

("sum revenue plus profit", ["plus"]),
("sum sales plus tax", ["plus"]),
("sum profit plus margin", ["plus"]),
("sum earnings plus bonus", ["plus"]),
("sum income plus interest", ["plus"]),

("total revenue plus profit", ["plus"]),
("total sales plus tax", ["plus"]),
("total profit plus margin", ["plus"]),
("total earnings plus bonus", ["plus"]),

("revenue + profit", ["plus"]),
("sales + tax", ["plus"]),
("profit + margin", ["plus"]),
("earnings + bonus", ["plus"]),
("income + interest", ["plus"]),

("add column revenue plus column profit", ["plus"]),
("add column sales plus column tax", ["plus"]),
("add column profit plus column margin", ["plus"]),

("compute revenue plus profit", ["plus"]),
("compute sales plus tax", ["plus"]),
("compute profit plus margin", ["plus"]),

("add revenue and profit together", ["plus"]),
("add sales and tax together", ["plus"]),
("add profit and margin together", ["plus"]),
# =====================================================
# EVEN MORE PLUS INTENT DATA
# =====================================================

# --- Natural addition phrases ---
("combine revenue and profit", ["plus"]),
("combine sales and tax", ["plus"]),
("combine profit and margin", ["plus"]),
("combine earnings and bonus", ["plus"]),
("combine income and interest", ["plus"]),

("merge revenue and profit values", ["plus"]),
("merge sales and tax values", ["plus"]),
("merge profit and margin values", ["plus"]),

("join revenue and profit together", ["plus"]),
("join sales and tax together", ["plus"]),
("join profit and margin together", ["plus"]),

# --- Spreadsheet style ---
("add the revenue column to the profit column", ["plus"]),
("add the sales column to the tax column", ["plus"]),
("add the profit column to the margin column", ["plus"]),
("sum the revenue and profit columns", ["plus"]),
("sum the sales and tax columns", ["plus"]),
("sum the profit and margin columns", ["plus"]),

# --- Formula style ---
("create revenue plus profit formula", ["plus"]),
("create sales plus tax formula", ["plus"]),
("create profit plus margin formula", ["plus"]),
("build revenue plus profit calculation", ["plus"]),
("build sales plus tax calculation", ["plus"]),

# --- Analytical phrasing ---
("calculate total revenue and profit", ["plus"]),
("calculate total sales and tax", ["plus"]),
("calculate total profit and margin", ["plus"]),
("calculate combined revenue and profit", ["plus"]),
("calculate combined sales and tax", ["plus"]),

# --- Business phrasing ---
("revenue combined with profit", ["plus"]),
("sales combined with tax", ["plus"]),
("profit combined with margin", ["plus"]),
("earnings combined with bonus", ["plus"]),

# --- Data analysis style ---
("aggregate revenue and profit", ["plus"]),
("aggregate sales and tax", ["plus"]),
("aggregate profit and margin", ["plus"]),

("add up revenue and profit", ["plus"]),
("add up sales and tax", ["plus"]),
("add up profit and margin", ["plus"]),

# --- Instruction style ---
("please add revenue and profit", ["plus"]),
("please add sales and tax", ["plus"]),
("please add profit and margin", ["plus"]),

("take revenue and add profit", ["plus"]),
("take sales and add tax", ["plus"]),
("take profit and add margin", ["plus"]),

# --- Builder phrasing ---
("add revenue and profit inside the formula", ["plus"]),
("add sales and tax inside the formula", ["plus"]),
("add profit and margin inside the formula", ["plus"]),

("insert revenue plus profit in the formula", ["plus"]),
("insert sales plus tax in the formula", ["plus"]),

# --- Column builder combinations ---
("open column builder and add revenue plus profit",
 ["open_column_builder","plus"]),

("open column builder and compute sales plus tax",
 ["open_column_builder","plus"]),

# --- Formula column combinations ---
("create a formula column using revenue plus profit",
 ["add_formula_column","plus"]),

("add a calculated column where sales plus tax",
 ["add_formula_column","plus"]),

("build a formula column using profit plus margin",
 ["add_formula_column","plus"]),

# --- Multi intent combinations ---
("rename the column and calculate revenue plus profit",
 ["set_new_column_name","plus"]),

("rename the column and compute sales plus tax",
 ["set_new_column_name","plus"]),

# --- Full workflow examples ---
("upload the file create a sheet called revenue summary add a formula column with revenue plus profit rename the column total revenue set month as x axis and profit as y axis and open column builder",
 ["upload_file","create_new_sheet","name_new_sheet","add_formula_column","plus","set_new_column_name","set_x_axis","set_y_axis","open_column_builder"]),

("import dataset create a new sheet called sales summary calculate sales plus tax rename the column total sales set quarter as x axis set growth as y axis and open column builder",
 ["upload_file","create_new_sheet","name_new_sheet","plus","set_new_column_name","set_x_axis","set_y_axis","open_column_builder"]),
 # =====================================================
# PLUS INTENT - ADDITION OPERATOR
# =====================================================

("add these two values", ["plus"]),
("add the numbers together", ["plus"]),
("add these fields", ["plus"]),
("add both columns", ["plus"]),
("add the two columns", ["plus"]),

("sum these values", ["plus"]),
("sum the columns", ["plus"]),
("sum these two numbers", ["plus"]),
("sum the two variables", ["plus"]),
("sum the selected fields", ["plus"]),

("combine these values", ["plus"]),
("combine the two columns", ["plus"]),
("combine the variables", ["plus"]),
("combine both numbers", ["plus"]),
("combine the selected fields", ["plus"]),

("total these values", ["plus"]),
("total the two numbers", ["plus"]),
("total the columns", ["plus"]),
("calculate the total", ["plus"]),
("calculate the sum", ["plus"]),

("add revenue and profit", ["plus"]),
("add sales and tax", ["plus"]),
("add income and bonus", ["plus"]),
("add cost and margin", ["plus"]),
("add price and fee", ["plus"]),

("revenue plus profit", ["plus"]),
("sales plus tax", ["plus"]),
("income plus bonus", ["plus"]),
("cost plus margin", ["plus"]),
("price plus fee", ["plus"]),

("revenue + profit", ["plus"]),
("sales + tax", ["plus"]),
("income + bonus", ["plus"]),
("cost + margin", ["plus"]),
("price + fee", ["plus"]),

("add the revenue column with profit", ["plus"]),
("add the sales column with tax", ["plus"]),
("add income with bonus", ["plus"]),
("add cost with margin", ["plus"]),
("add price with fee", ["plus"]),

("include revenue and profit in the total", ["plus"]),
("include sales and tax in the total", ["plus"]),
("include income and bonus together", ["plus"]),

("perform addition", ["plus"]),
("do addition", ["plus"]),
("use addition", ["plus"]),
("apply addition", ["plus"]),

("insert a plus operator", ["plus"]),
("use the plus operator", ["plus"]),
("place a plus sign", ["plus"]),
("add a plus symbol", ["plus"]),
("put a plus between the variables", ["plus"]),

("add the selected variables", ["plus"]),
("add the chosen fields", ["plus"]),
("add these inputs", ["plus"]),
("add the selected columns together", ["plus"]),

("calculate revenue plus profit", ["plus"]),
("calculate sales plus tax", ["plus"]),
("calculate income plus bonus", ["plus"]),
("calculate cost plus margin", ["plus"]),
("calculate price plus fee", ["plus"]),
# =====================================================
# MINUS INTENT - SUBTRACTION OPERATOR
# =====================================================

("subtract these values", ["minus"]),
("subtract the numbers", ["minus"]),
("subtract these fields", ["minus"]),
("subtract the columns", ["minus"]),
("subtract the two values", ["minus"]),

("deduct this value", ["minus"]),
("deduct the cost", ["minus"]),
("deduct tax from revenue", ["minus"]),
("deduct the fee", ["minus"]),
("deduct the expense", ["minus"]),

("remove this value from the total", ["minus"]),
("remove the cost from revenue", ["minus"]),
("remove tax from sales", ["minus"]),
("remove the fee from the price", ["minus"]),
("remove the expense from income", ["minus"]),

("calculate the difference", ["minus"]),
("calculate the difference between values", ["minus"]),
("find the difference between numbers", ["minus"]),
("compute the difference", ["minus"]),
("get the difference between the columns", ["minus"]),

("revenue minus cost", ["minus"]),
("sales minus tax", ["minus"]),
("income minus expenses", ["minus"]),
("profit minus loss", ["minus"]),
("price minus discount", ["minus"]),

("revenue - cost", ["minus"]),
("sales - tax", ["minus"]),
("income - expenses", ["minus"]),
("profit - loss", ["minus"]),
("price - discount", ["minus"]),

("subtract cost from revenue", ["minus"]),
("subtract tax from sales", ["minus"]),
("subtract expenses from income", ["minus"]),
("subtract discount from price", ["minus"]),

("less tax", ["minus"]),
("less cost", ["minus"]),
("less expenses", ["minus"]),
("less discount", ["minus"]),

("apply subtraction", ["minus"]),
("perform subtraction", ["minus"]),
("use subtraction", ["minus"]),
("do subtraction", ["minus"]),

("insert a minus operator", ["minus"]),
("use the minus operator", ["minus"]),
("place a minus sign", ["minus"]),
("add a minus symbol", ["minus"]),
("put a minus between the variables", ["minus"]),

("subtract the selected values", ["minus"]),
("subtract the selected columns", ["minus"]),
("subtract the selected variables", ["minus"]),

("calculate revenue minus cost", ["minus"]),
("calculate sales minus tax", ["minus"]),
("calculate income minus expenses", ["minus"]),
("calculate profit minus loss", ["minus"]),
("calculate price minus discount", ["minus"])

]





# =====================================================
# BUILD INTENT MAP
# =====================================================

all_intents = sorted(list(set(i for _, intents in intent_data for i in intents)))

intent_to_idx = {intent: i for i, intent in enumerate(all_intents)}
idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = SimpleTokenizer()
tokenizer.fit([text for text, _ in intent_data])

# =====================================================
# DATASET
# =====================================================

X = []
Y = []

for text, intents in intent_data:

    encoded = tokenizer.encode(text)

    if len(encoded) < MAX_LEN:
        encoded = encoded + [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    X.append(encoded)

    label = np.zeros(len(all_intents))

    for intent in intents:
        label[intent_to_idx[intent]] = 1

    Y.append(label)

X = torch.tensor(np.array(X), dtype=torch.long)
Y = torch.tensor(np.array(Y), dtype=torch.float32)

# =====================================================
# MODEL CONFIG (IMPORTANT FOR APP LOADING)
# =====================================================

MODEL_CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "num_classes": len(all_intents)
}

# =====================================================
# MODEL
# =====================================================

model = TransformerIntentClassifier(
    vocab_size=MODEL_CONFIG["vocab_size"],
    num_classes=MODEL_CONFIG["num_classes"]
)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================
# TRAIN
# =====================================================

print("Training model...")

epochs = 500

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, Y)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

print("Training complete!")

# =====================================================
# SAVE MODEL
# =====================================================

checkpoint = {
    "model_state": model.state_dict(),
    "word2idx": tokenizer.word2idx,
    "idx_to_intent": idx_to_intent,
    "intent_to_idx": intent_to_idx,
    "max_len": MAX_LEN,
    "model_config": MODEL_CONFIG
}

torch.save(checkpoint, MODEL_PATH)

print("Model saved ->", MODEL_PATH)