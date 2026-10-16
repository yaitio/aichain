"""The same agent as two serverless handlers, sharing nothing but data/.

The signature is AWS Lambda's and nothing imports AWS: deployed, the store
becomes DynamoDB or S3 and the event comes from API Gateway or a queue, and
desk.py does not change.
"""

from yait_aichain.state import RunContext, SuspendedResult, using

from desk import build_agent


def _reply(result) -> dict:
    return {"status": "answered" if result.success else "failed", "reply": result.output,
            "stopped_by": result.stopped_by, "tokens": result.tokens_used, "cost": result.cost}


def start(event: dict, context=None) -> dict:
    """{"tenant", "message"} → {"status": "answered" | "parked", "run_id", "reply", …}"""
    with using(RunContext(tenant=event["tenant"])):
        result = build_agent().run(event["message"])
    if isinstance(result, SuspendedResult):
        return {"status": "parked", "run_id": result.run_id,
                "reply": result.awaiting.get("reason"), "stopped_by": None}
    return {"run_id": None, **_reply(result)}


def approve(event: dict, context=None) -> dict:
    """{"tenant", "run_id", "approved", "note"} → {"status", "reply", …}"""
    with using(RunContext(tenant=event["tenant"])):
        result = build_agent().resume(event["run_id"], signal={"approved": event["approved"],
                                                               "note": event.get("note", "")})
    return _reply(result)
