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