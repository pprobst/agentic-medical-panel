#!/usr/bin/env python3

import os
import argparse
import time
from dataclasses import dataclass, field

import asyncio

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from dotenv import load_dotenv

from utils.log import log
from utils.template_manager import TemplateManager

_ = load_dotenv()

# --------------------------------------------------------------------------
# 1. Configuration and Dependencies
# --------------------------------------------------------------------------

if os.getenv("LOGFIRE_TOKEN"):
    _ = logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    _ = logfire.instrument_pydantic_ai()


def resolve_model_string(model_name: str) -> str:
    """Convert a model name to a pydantic-ai model string."""
    if model_name.startswith("gemini"):
        return f"google-gla:{model_name}"
    return f"openai:{model_name}"


@dataclass
class Dependencies:
    """Shared dependencies injected into all agents."""

    model: str
    model_settings: OpenAIChatModelSettings | None
    template_manager: TemplateManager = field(repr=False)


# --------------------------------------------------------------------------
# 2. Define Core Data Structures
# --------------------------------------------------------------------------


class Diagnosis(BaseModel):
    condition: str = Field(..., description="The medical condition being diagnosed.")
    probability: float = Field(
        ..., ge=0, le=1, description="The estimated probability of this diagnosis."
    )
    reasoning: str = Field(
        ..., description="Justification for this diagnosis based on patient data."
    )


class TestRequest(BaseModel):
    test_name: str = Field(..., description="The specific name of the medical test.")
    reasoning: str = Field(
        ...,
        description="Why this test will maximally discriminate between the leading hypotheses.",
    )


class StewardshipAdvice(BaseModel):
    test_name: str = Field(..., description="The test being evaluated.")
    is_approved: bool = Field(
        ..., description="Whether the test is approved from a cost perspective."
    )
    justification: str = Field(..., description="Reasoning for the approval or veto.")


class ChallengerCritique(BaseModel):
    identified_bias: str = Field(
        ..., description="The primary cognitive bias identified (e.g., anchoring)."
    )
    contradictory_evidence: str = Field(
        ...,
        description="Evidence from the case that contradicts the leading hypothesis.",
    )
    falsification_test_suggestion: TestRequest = Field(
        ..., description="A test designed to falsify the leading diagnosis."
    )


class QualityCheck(BaseModel):
    check_name: str = Field(..., description="The name of the quality check performed.")
    is_consistent: bool = Field(
        ..., description="Whether the panel's reasoning is internally consistent."
    )
    comment: str = Field(..., description="Comments on any identified inconsistencies.")


class DebateResults(BaseModel):
    challenger_critique: ChallengerCritique = Field(
        ..., description="The critique from the challenger."
    )
    stewardship_advice: list[StewardshipAdvice] = Field(
        ..., description="The advice from the stewardship panel."
    )
    quality_checks: list[QualityCheck] = Field(
        ..., description="The quality checks from the checklist panel."
    )


class FinalDecision(BaseModel):
    action: Diagnosis | TestRequest = Field(
        ..., description="The final chosen action after consensus."
    )
    consensus_summary: str = Field(
        ...,
        description="A summary of the debate and justification for the final action.",
    )


# --------------------------------------------------------------------------
# 3. Define the Orchestrator Agent and its Tools
# --------------------------------------------------------------------------

orchestrator_agent = Agent(
    output_type=FinalDecision,
    deps_type=Dependencies,
    instructions="""\
You are the orchestrator of a virtual medical panel. Your goal is to determine the most likely diagnosis or the next best action for a patient.

Follow this exact process:
1.  **Hypothesize**: Use the `hypothesize` tool to generate initial diagnoses.
2.  **Check Certainty**: If the top hypothesis probability is >= 95%, formulate a `FinalDecision` to commit to it and you are done.
3.  **Propose Tests**: If certainty is < 95%, use the `request_tests` tool to propose tests.
4.  **Debate**: Use the `debate` tool to gather critiques of the current plan.
5.  **Synthesize**: **You must call the `reach_consensus` tool.** Pass all the information gathered so far (patient info, hypotheses, tests, and debate results) to it. This tool will provide the final, synthesized action.
6.  **Finalize**: Package the action from `reach_consensus` into the `FinalDecision` object, adding a brief summary. Your job is to orchestrate, not to decide the final action yourself.
""",
)


@orchestrator_agent.tool
async def hypothesize(
    ctx: RunContext[Dependencies], patient_info: str
) -> list[Diagnosis]:
    """Generates a ranked list of the top 3 potential diagnoses."""
    deps = ctx.deps
    prompt = deps.template_manager.render(
        "dr_hypothesis.jinja2", patient_info=patient_info
    )
    log.info("Tool: Running Dr. Hypothesis...")
    hypothesis_agent = Agent(
        deps.model,
        output_type=list[Diagnosis],
        model_settings=deps.model_settings,
        tools=[duckduckgo_search_tool()],
    )
    result = await hypothesis_agent.run(prompt, usage=ctx.usage)
    log.info(
        f"Dr. Hypothesis's Differential Diagnosis: {[h.condition for h in result.output]}"
    )
    return result.output


@orchestrator_agent.tool
async def request_tests(
    ctx: RunContext[Dependencies], patient_info: str, hypotheses: list[Diagnosis]
) -> list[TestRequest]:
    """Proposes up to 3 diagnostic tests to differentiate between the top hypotheses."""
    deps = ctx.deps
    prompt = deps.template_manager.render(
        "dr_test_chooser.jinja2", hypotheses=hypotheses, patient_info=patient_info
    )
    log.info("Tool: Running Dr. Test Chooser...")
    test_chooser_agent = Agent(
        deps.model,
        output_type=list[TestRequest],
        model_settings=deps.model_settings,
    )
    result = await test_chooser_agent.run(prompt, usage=ctx.usage)
    log.info(f"Dr. Test Chooser recommends: {[t.test_name for t in result.output]}")
    return result.output


@orchestrator_agent.tool
async def debate(
    ctx: RunContext[Dependencies],
    hypotheses: list[Diagnosis],
    test_requests: list[TestRequest],
) -> DebateResults:
    """Runs the deliberation panel to critique the current plan."""
    deps = ctx.deps
    log.info("Tool: Convening the debate panel...")

    challenger_agent = Agent(
        deps.model,
        output_type=ChallengerCritique,
        model_settings=deps.model_settings,
    )
    stewardship_agent = Agent(
        deps.model,
        output_type=list[StewardshipAdvice],
        model_settings=deps.model_settings,
        tools=[duckduckgo_search_tool()],
    )
    checklist_agent = Agent(
        deps.model,
        output_type=list[QualityCheck],
        model_settings=deps.model_settings,
    )

    challenger_result, stewardship_result, checklist_result = await asyncio.gather(
        challenger_agent.run(
            deps.template_manager.render(
                "dr_challenger.jinja2",
                hypotheses=hypotheses,
                test_requests=test_requests,
            ),
            usage=ctx.usage,
        ),
        stewardship_agent.run(
            deps.template_manager.render(
                "dr_stewardship.jinja2", test_requests=test_requests
            ),
            usage=ctx.usage,
        ),
        checklist_agent.run(
            deps.template_manager.render(
                "dr_checklist.jinja2",
                hypotheses=hypotheses,
                test_requests=test_requests,
            ),
            usage=ctx.usage,
        ),
    )

    critique = challenger_result.output
    advice = stewardship_result.output
    checks = checklist_result.output

    log.info(f"Dr. Challenger found bias: {critique.identified_bias}")
    log.info(f"Dr. Challenger's critique:\n{critique.model_dump_json(indent=2)}")
    log.info(
        f"Dr. Stewardship approved {sum(1 for a in advice if a.is_approved)}/{len(advice)} tests."
    )
    log.info(
        "Dr. Stewardship's advice:\n"
        + "\n".join(a.model_dump_json(indent=2) for a in advice)
    )
    log.info(
        f"Dr. Checklist passed {sum(1 for c in checks if c.is_consistent)}/{len(checks)} checks."
    )
    log.info(
        "Dr. Checklist's checks:\n"
        + "\n".join(c.model_dump_json(indent=2) for c in checks)
    )

    return DebateResults(
        challenger_critique=critique,
        stewardship_advice=advice,
        quality_checks=checks,
    )


@orchestrator_agent.tool
async def reach_consensus(
    ctx: RunContext[Dependencies],
    patient_info: str,
    hypotheses: list[Diagnosis],
    test_requests: list[TestRequest],
    debate_results: DebateResults,
) -> Diagnosis | TestRequest:
    """Synthesizes all information using the decision-maker prompt to select the single best action."""
    deps = ctx.deps
    log.info("Tool: Running Consensus Panel to make final decision...")

    prompt = deps.template_manager.render(
        "dr_decision_maker.jinja2",
        patient_info=patient_info,
        hypotheses=hypotheses,
        test_requests=test_requests,
        challenger_critique=debate_results.challenger_critique,
        stewardship_advice=debate_results.stewardship_advice,
        quality_checks=debate_results.quality_checks,
    )
    consensus_agent = Agent(
        deps.model,
        output_type=Diagnosis | TestRequest,
        model_settings=deps.model_settings,
    )
    result = await consensus_agent.run(prompt, usage=ctx.usage)
    log.info(f"Consensus panel decided on action: {type(result.output).__name__}")
    return result.output


# --------------------------------------------------------------------------
# 4. Main Execution Block
# --------------------------------------------------------------------------


async def main() -> None:
    """Runs a single pass of the diagnostic orchestrator."""
    parser = argparse.ArgumentParser(description="Run the diagnostic orchestrator.")
    _ = parser.add_argument(
        "--patient-info",
        type=str,
        default="A 65-year-old male presents with a 3-day history of high fever, a productive cough, and shortness of breath.",
        help="The patient case information.",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="The model to use for each agent. Default is gpt-4.1-mini.",
    )
    args = parser.parse_args()

    model_name = str(args.model)
    patient_info = str(args.patient_info)

    model_string = resolve_model_string(model_name)
    model_settings = (
        OpenAIChatModelSettings(openai_reasoning_effort="low")
        if "gpt-5" in model_name
        else None
    )
    deps = Dependencies(
        model=model_string,
        model_settings=model_settings,
        template_manager=TemplateManager("templates"),
    )

    log.info("--- Starting Diagnostic Process ---")
    log.info(f"Patient Info: {patient_info}")
    log.info(f"Model: {model_string}")

    start_time = time.time()
    result = await orchestrator_agent.run(
        patient_info,
        deps=deps,
        model=deps.model,
        model_settings=deps.model_settings,
    )
    total_duration = time.time() - start_time
    log.info(f"--- Orchestrator Conclusion (Total time: {total_duration:.2f}s) ---")
    log.info(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
