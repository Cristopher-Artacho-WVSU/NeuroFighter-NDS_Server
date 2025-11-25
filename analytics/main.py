from enum import Enum
from typing import Literal, cast
from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from polars import DataFrame
import polars as pl
import altair as alt

from .data_cleanup import prepare_history, reshape_df, rules_to_df, data_processing_queue
analytics_router = APIRouter(
    prefix = "/analytics/v0",
)

analytics_templates = Jinja2Templates(directory="analytics/templates")
rules_adjustments_history: list[list[dict[Literal["rule_id", "weight_adjustment"], int | float]]] = []
last_rule: list[dict[Literal["rule_id", "weight_adjustment"], int | float]] = []
rule_usage: dict[int, float] = {}
rules_history_df: DataFrame = DataFrame()

@analytics_router.get("/tables", response_class=HTMLResponse)
async def weight_changes_page(request: Request):
    return analytics_templates.TemplateResponse(
        request=request, name="tables.html"
    )

@analytics_router.patch("/tables", response_class=HTMLResponse)
async def weight_changes(request: Request):
    global last_rule, rule_usage
    try:
        if last_rule != rules_adjustments_history[-1]:
            for rule_adjusted in last_rule:
                if rule_adjusted['rule_id'] in rule_usage:
                    rule_usage[cast(int, rule_adjusted['rule_id'])] += 1
                else:
                    rule_usage[cast(int, rule_adjusted['rule_id'])] = 1
        last_rule = rules_adjustments_history[-1]
    except IndexError as _:
        return """
            <template>
                <tr id="rule-adjustment" hx-patch="/analytics/v0/tables" hx-trigger="every 500ms" hx-swap-oob="true"></tr>
            </template>
        """
    except Exception as _:
        return HTTPException(status_code=500, detail="Something went wrong when trying to use rule adjustment data.")
    return analytics_templates.TemplateResponse(
        request=request, name="table_content.html", context={"rules_adjusted": last_rule, "rules_used": rule_usage}
    )

@analytics_router.get("/history", response_class=HTMLResponse)
async def weight_history_display(request: Request):
    return analytics_templates.TemplateResponse(
        request=request, name="weight-history.html"
    )

class AnalyticsType(str, Enum):
    weights = "weights"
    usage = "usage"

@analytics_router.patch("/history/{analytics}", response_class=HTMLResponse)
async def weight_history(analytics: AnalyticsType):
    global rules_history_df
    script_rules_df = await rules_to_df(data_processing_queue)
    if not rules_history_df.is_empty():
        rules_history_df = pl.concat([rules_history_df, script_rules_df.select(['ruleID', 'weight', 'wasUsed']).with_columns(pl.lit(rules_history_df.shape[0]).alias("running_count"))], how='vertical_relaxed')
    else:
        rules_history_df = script_rules_df.select(['ruleID', 'weight', 'wasUsed']).with_columns(pl.lit(rules_history_df.shape[0]).alias("running_count"))

    rules_history_reshaped = await reshape_df(rules_history_df)
    rules_history_for_graph = await prepare_history(rules_history_reshaped)
    weight_history = rules_history_for_graph.select(
        pl.col("^.*weight.*$", "running_count")).unpivot(
        index="running_count",
        variable_name="item_id",
        value_name="weight_value"
    ).with_columns(
        pl.col("item_id").str.extract(r'(\d+)', 1).alias("Item_ID")
    ).drop("item_id")
    usage = rules_history_for_graph.select(
        pl.col("^.*wasUsed.*$", "running_count")).unpivot(
        index="running_count",
        variable_name="item_id",
        value_name="was_used"
    ).with_columns(
        pl.col("item_id").str.extract(r'(\d+)', group_index=1).alias("Item_ID")
    ).cast({"was_used": pl.Boolean}).drop("item_id")
    match analytics:
        case "weights":
            weight_history_chart: str = alt.Chart(weight_history).mark_line().encode(
                alt.X("running_count"),
                alt.Y(
                    "weight_value"
                ).scale(
                    domain=[0,1]
                ),
                alt.Color("Item_ID")
            ).properties(
                width=800,
                height=300
            ).to_html(output_div="weight-update-graphs", fullhtml=False)
            return HTMLResponse(weight_history_chart)
        case "usage":
            print(usage)
            ...
    
def clear_history():
    global rules_history_df
    rules_history_df = DataFrame()
