#!/usr/bin/env python3

import os
import argparse
import time

import asyncio

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool  # pyright: ignore[reportUnknownVariableType]
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


class Config:
    """Application configuration."""

    def __init__(self, model: str) -> None:
        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.model: OpenAIChatModel = OpenAIChatModel(model)
        self.model_settings: OpenAIChatModelSettings | None = (
            OpenAIChatModelSettings(
                openai_reasoning_effort="low",
            )
            if "gpt-5" in self.model.model_name
            else None
        )
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.template_dir: str = "templates"


class Dependencies:
    """Container for shared dependencies."""

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.template_manager: TemplateManager = TemplateManager(config.template_dir)


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
    model=OpenAIChatModel("gpt-4.1-mini"),
    output_type=FinalDecision,
    deps_type=Dependencies,
    system_prompt="""
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
    context: RunContext[Dependencies], patient_info: str
) -> list[Diagnosis]:
    """Generates a ranked list of the top 3 potential diagnoses."""
    deps = context.deps

    try:
        prompt = deps.template_manager.render(
            "dr_hypothesis.jinja2", patient_info=patient_info
        )
        log.info("Tool: Running Dr. Hypothesis...")
        hypothesis_agent = Agent(
            output_type=list[Diagnosis],
            model=deps.config.model,
            model_settings=deps.config.model_settings,
            tools=[duckduckgo_search_tool()],
        )
        result = await hypothesis_agent.run(prompt)
        log.info(
            f"Dr. Hypothesis's Differential Diagnosis: {[h.condition for h in result.output]}"
        )
        return result.output
    except Exception as e:
        log.error(f"Error in hypothesize tool: {str(e)}")
        raise


@orchestrator_agent.tool
async def request_tests(
    context: RunContext[Dependencies], patient_info: str, hypotheses: list[Diagnosis]
) -> list[TestRequest]:
    """Proposes up to 3 diagnostic tests to differentiate between the top hypotheses."""
    deps = context.deps

    try:
        prompt = deps.template_manager.render(
            "dr_test_chooser.jinja2", hypotheses=hypotheses, patient_info=patient_info
        )
        log.info("Tool: Running Dr. Test Chooser...")
        test_chooser_agent = Agent(
            output_type=list[TestRequest],
            model=deps.config.model,
            model_settings=deps.config.model_settings,
        )
        result = await test_chooser_agent.run(prompt)
        log.info(f"Dr. Test Chooser recommends: {[t.test_name for t in result.output]}")
        return result.output
    except Exception as e:
        log.error(f"Error in request_tests tool: {str(e)}")
        raise


@orchestrator_agent.tool
async def debate(
    context: RunContext[Dependencies],
    hypotheses: list[Diagnosis],
    test_requests: list[TestRequest],
) -> DebateResults:
    """Runs the deliberation panel to critique the current plan."""
    deps = context.deps
    log.info("Tool: Convening the debate panel...")

    async def run_challenger():
        try:
            prompt = deps.template_manager.render(
                "dr_challenger.jinja2",
                hypotheses=hypotheses,
                test_requests=test_requests,
            )
            challenger_agent = Agent(
                output_type=ChallengerCritique,
                model=deps.config.model,
                model_settings=deps.config.model_settings,
            )
            return await challenger_agent.run(prompt)
        except Exception as e:
            log.error(f"Error in challenger: {str(e)}")
            raise

    async def run_stewardship():
        try:
            prompt = deps.template_manager.render(
                "dr_stewardship.jinja2", test_requests=test_requests
            )
            stewardship_agent = Agent(
                output_type=list[StewardshipAdvice],
                model=deps.config.model,
                model_settings=deps.config.model_settings,
                tools=[duckduckgo_search_tool()],
            )
            return await stewardship_agent.run(prompt)
        except Exception as e:
            log.error(f"Error in stewardship: {str(e)}")
            raise

    async def run_checklist():
        try:
            prompt = deps.template_manager.render(
                "dr_checklist.jinja2",
                hypotheses=hypotheses,
                test_requests=test_requests,
            )
            checklist_agent = Agent(
                output_type=list[QualityCheck],
                model=deps.config.model,
                model_settings=deps.config.model_settings,
            )
            return await checklist_agent.run(prompt)
        except Exception as e:
            log.error(f"Error in checklist: {str(e)}")
            raise

    try:
        results = await asyncio.gather(
            run_challenger(), run_stewardship(), run_checklist()
        )
        critique, advice, checks = (
            results[0].output,
            results[1].output,
            results[2].output,
        )
        critique_dump = critique.model_dump_json(indent=2)
        log.info(f"Dr. Challenger found bias: {critique.identified_bias}")
        log.info(f"Dr. Challenger's critique:\n{critique_dump}")
        log.info(
            f"Dr. Stewardship approved {sum(1 for a in advice if a.is_approved)}/{len(advice)} tests."
        )
        advice_dump = "\n".join([a.model_dump_json(indent=2) for a in advice])
        log.info(f"Dr. Stewardship's advice:\n{advice_dump}")
        log.info(
            f"Dr. Checklist passed {sum(1 for c in checks if c.is_consistent)}/{len(checks)} checks."
        )
        checks_dump = "\n".join([c.model_dump_json(indent=2) for c in checks])
        log.info(f"Dr. Checklist's checks:\n{checks_dump}")
        return DebateResults(
            challenger_critique=critique,
            stewardship_advice=advice,
            quality_checks=checks,
        )
    except Exception as e:
        log.error(f"Error in debate tool: {str(e)}")
        raise


@orchestrator_agent.tool
async def reach_consensus(
    context: RunContext[Dependencies],
    patient_info: str,
    hypotheses: list[Diagnosis],
    test_requests: list[TestRequest],
    debate_results: DebateResults,
) -> Diagnosis | TestRequest:
    """Synthesizes all information using the decision-maker prompt to select the single best action."""
    deps = context.deps
    log.info("Tool: Running Consensus Panel to make final decision...")

    try:
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
            output_type=Diagnosis | TestRequest,
            model=deps.config.model,
            model_settings=deps.config.model_settings,
        )
        result = await consensus_agent.run(prompt)
        log.info(f"Consensus panel decided on action: {type(result.output).__name__}")
        return result.output
    except Exception as e:
        log.error(f"Error in reach_consensus tool: {str(e)}")
        raise


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

    patient_case_info = args.patient_info
    model_name = args.model

    # Initialize configuration and dependencies
    config = Config(model=model_name)
    try:
        deps = Dependencies(config)
    except Exception as e:
        log.error(f"Error initializing dependencies: {e}")
        log.error(
            "Please ensure you have a 'templates' directory at the project root with the required .jinja2 files."
        )
        return

    log.info("--- Starting Diagnostic Process ---")
    log.info(f"Patient Info: {patient_case_info}")
    log.info(f"Model: {config.model.model_name}")
    log.info(f"Template Directory: {config.template_dir}")

    # Run orchestrator with dependencies
    start_time = time.time()
    try:
        final_decision_result = await orchestrator_agent.run(
            patient_case_info, deps=deps
        )
        total_duration = time.time() - start_time
        log.info(f"--- Orchestrator Conclusion (Total time: {total_duration:.2f}s) ---")
        log.info(final_decision_result.output.model_dump_json(indent=2))
    except Exception as e:
        total_duration = time.time() - start_time
        log.error(f"Diagnostic process failed after {total_duration:.2f}s: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
