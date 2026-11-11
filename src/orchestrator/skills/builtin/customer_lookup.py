"""Look up the customer's account profile and recent support history (simulated CRM)."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field

from orchestrator.contracts import Skill, SkillContext

_CUSTOMER_ID = re.compile(r"\bCUST-\d{3,8}\b", re.IGNORECASE)
_PLANS = ["free", "pro", "business", "enterprise"]
_REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]


class CustomerInput(BaseModel):
    text: str = Field(default="", description="Message to scan for a customer id")
    customer_id: str | None = None


class CustomerOutput(BaseModel):
    found: bool
    customer_id: str | None
    plan: str | None
    region: str | None
    open_tickets: int = Field(ge=0)
    tickets_last_90_days: int = Field(ge=0)


class CustomerLookup(Skill[CustomerInput, CustomerOutput]):
    name = "customer_lookup"
    description = "Resolve a CUST-nnnn id to plan, region and ticket history (deterministic CRM stub)."
    input_model = CustomerInput
    output_model = CustomerOutput

    async def run(self, args: CustomerInput, ctx: SkillContext) -> CustomerOutput:
        cid = args.customer_id
        if not cid:
            m = _CUSTOMER_ID.search(args.text)
            cid = m.group(0).upper() if m else None
        if not cid:
            return CustomerOutput(
                found=False, customer_id=None, plan=None, region=None, open_tickets=0, tickets_last_90_days=0
            )
        h = int(hashlib.sha256(cid.encode()).hexdigest(), 16)
        return CustomerOutput(
            found=True,
            customer_id=cid,
            plan=_PLANS[h % len(_PLANS)],
            region=_REGIONS[(h >> 8) % len(_REGIONS)],
            open_tickets=(h >> 16) % 3,
            tickets_last_90_days=(h >> 24) % 9,
        )
