"""
OpenAI Responses API client with function calling for get_weather.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from weather_service import get_weather

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("weather_chat.audit")

MODEL = "gpt-5-nano"

INSTRUCTIONS = """You are a helpful assistant that answers weather questions using the get_weather tool.
- For any question about temperature, wind, humidity, rain, or conditions in a city, call get_weather with the city name.
- The tool's date parameter must be ISO YYYY-MM-DD. Convert phrases like "January 1st 2020" or "4th of July 2026" into that form. If the user omits the year, infer the most reasonable year from context or the current date.
- Never invent numeric weather values; only state numbers that appear in tool results (or say the tool failed).
- If the tool returns source \"climate_projection\", clearly tell the user these are long-range climate-model projections with large uncertainty, not a reliable day-ahead forecast. Include the disclaimer tone from the tool when appropriate.
- If the tool returns an error field, explain it briefly and suggest a fix (e.g. spelling, adding country, or a valid date).
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get current weather or daily summary for a city. "
            "Use date in YYYY-MM-DD for a specific calendar day (past, today, near future, or far future). "
            "Omit date for live current conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name; add region or country if ambiguous.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional. ISO date YYYY-MM-DD for that day's summary.",
                },
            },
            "required": ["city"],
        },
    },
]


def _response_final_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "output_text":
                parts.append(getattr(block, "text", "") or "")
            elif btype == "text":
                parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _log_tool_audit(
    *,
    user_query: str,
    tool_name: str,
    llm_tool_parameters: Any,
    weather_service_response: Any,
    llm_arguments_raw: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "user_query": user_query,
        "tool": tool_name,
        "llm_tool_parameters": llm_tool_parameters,
        "weather_service_response": weather_service_response,
    }
    if llm_arguments_raw is not None:
        record["llm_arguments_raw"] = llm_arguments_raw
    if extra:
        record.update(extra)
    audit_logger.info(json.dumps(record, ensure_ascii=False))


def _execute_get_weather(
    arguments: str | dict[str, Any] | None,
    *,
    user_query: str,
) -> str:
    if arguments is None:
        payload = get_weather("")
        _log_tool_audit(
            user_query=user_query,
            tool_name="get_weather",
            llm_tool_parameters=None,
            weather_service_response=payload,
            extra={"note": "model sent null arguments; called get_weather(\"\")"},
        )
        return json.dumps(payload)

    raw: Any
    if isinstance(arguments, dict):
        raw = arguments
    else:
        try:
            raw = json.loads(arguments)
        except json.JSONDecodeError as e:
            err = {
                "error": "Invalid tool arguments JSON",
                "detail": str(e),
            }
            _log_tool_audit(
                user_query=user_query,
                tool_name="get_weather",
                llm_tool_parameters=None,
                weather_service_response=err,
                llm_arguments_raw=arguments,
                extra={"parse_error": str(e)},
            )
            return json.dumps(err)

    if not isinstance(raw, dict):
        err = {"error": "Tool arguments must be a JSON object"}
        _log_tool_audit(
            user_query=user_query,
            tool_name="get_weather",
            llm_tool_parameters=raw,
            weather_service_response=err,
        )
        return json.dumps(err)

    city = raw.get("city")
    date_val = raw.get("date")
    if date_val is not None and date_val != "" and not isinstance(date_val, str):
        date_val = str(date_val)
    if not isinstance(city, str):
        err = {"error": "city must be a string"}
        _log_tool_audit(
            user_query=user_query,
            tool_name="get_weather",
            llm_tool_parameters=dict(raw),
            weather_service_response=err,
        )
        return json.dumps(err)

    params = {"city": city, "date": date_val if date_val else None}
    result = get_weather(city, date_val if date_val else None)
    _log_tool_audit(
        user_query=user_query,
        tool_name="get_weather",
        llm_tool_parameters=params,
        weather_service_response=result,
    )
    return json.dumps(result)


def run_turn(client: OpenAI, conversation: list[dict[str, Any]], user_text: str) -> str:
    """
    Append user message, run tool loop until model returns text, return assistant text.
    Mutates conversation with all request/response items for multi-turn history.
    """
    conversation.append({"role": "user", "content": user_text})

    try:
        while True:
            response = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                tools=TOOLS,
                input=conversation,
            )
            output_items = list(response.output)
            conversation.extend(_output_to_input_dicts(output_items))

            function_calls = [
                i for i in output_items if getattr(i, "type", None) == "function_call"
            ]
            if not function_calls:
                return _response_final_text(response)

            for fc in function_calls:
                name = getattr(fc, "name", None)
                call_id = getattr(fc, "call_id", None)
                args = getattr(fc, "arguments", None)
                if not call_id:
                    logger.warning("function_call missing call_id: %s", fc)
                    continue
                if name == "get_weather":
                    out = _execute_get_weather(args, user_query=user_text)
                else:
                    out = json.dumps(
                        {
                            "error": "Unknown function",
                            "detail": str(name),
                        }
                    )
                    _log_tool_audit(
                        user_query=user_text,
                        tool_name=str(name),
                        llm_tool_parameters={"name": name, "arguments": args},
                        weather_service_response=json.loads(out),
                    )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out,
                    }
                )
    except RateLimitError as e:
        logger.exception("OpenAI rate limit")
        raise RuntimeError("The model service is rate-limited. Please wait and try again.") from e
    except APIConnectionError as e:
        logger.exception("OpenAI connection error")
        raise RuntimeError("Could not reach the model service. Check your network connection.") from e
    except APIError as e:
        logger.exception("OpenAI API error")
        raise RuntimeError(f"Model API error: {e}") from e


def _output_to_input_dicts(items: list[Any]) -> list[dict[str, Any]]:
    """Convert SDK output objects to JSON-serializable input dicts for the next request."""
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif hasattr(item, "model_dump"):
            out.append(item.model_dump(mode="json", exclude_none=True))
        else:
            raise TypeError(f"Unexpected Responses output item type: {type(item)!r}")
    return out
