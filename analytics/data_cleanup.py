from asyncio import Queue
from typing import Any, cast
import polars as pl
from polars import DataFrame, json_normalize
from json import loads

data_processing_queue: Queue[str] = Queue()
def fix_json_format(rules_from_script: str) -> str:
    return f"[{rules_from_script}]"

async def rules_to_df(rules_queue: Queue[str]) -> DataFrame:
    rule_json = await rules_queue.get()
    rules_this_cycle  = loads(rule_json)
    cycles_df = json_normalize(rules_this_cycle)
    rules_df = json_normalize(cycles_df.item(row=0, column="script"))
    return rules_df

async def reshape_df(rules_history: DataFrame) -> DataFrame:
    rules_history_long = rules_history.unpivot(on=['weight', 'wasUsed'], index=['ruleID', 'running_count'])
    rules_history_ordered = rules_history_long.pivot(index='running_count', on=['ruleID', 'variable'], values="value", aggregate_function='first')
    return rules_history_ordered

async def prepare_history(rules_history: DataFrame) -> DataFrame:
    return rules_history.fill_null(strategy='forward').fill_null(strategy='backward')
