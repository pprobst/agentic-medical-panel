# Agentic Medical Diagnosis Panel

This project implements a multi-agent system using [`pydantic-ai`](https://ai.pydantic.dev/) to simulate a medical diagnostic process. It is heavily inspired by the concepts outlined in Microsoft's research on AI-driven diagnostics: [The Path to Medical Superintelligence](https://microsoft.ai/new/the-path-to-medical-superintelligence/).

It features an orchestrator agent that coordinates a panel of specialized AI agents to:
1.  Generate initial differential diagnoses for a patient case.
2.  Propose diagnostic tests if the initial hypotheses are uncertain.
3.  Debate the proposed plan from multiple perspectives (identifying cognitive bias, checking for cost stewardship, and ensuring quality).
4.  Reach a final consensus on the most likely diagnosis or the next best test to perform.

## Setup

### Prerequisites

- Python 3.12 or higher
- uv (recommended)
- An OpenAI API key

### Installation

```bash
git clone https://github.com/pprobst/agentic-medical-panel.git
cd ai-playground/pydantic-ai-exp
uv sync
source .venv/bin/activate
```

## Usage

Run the main application with patient information:

```bash
uv run main.py --patient-info "A 65-year-old male presents with a 3-day history of high fever, a productive cough, and shortness of breath." --model gpt-4.1-mini
```

### Example Output

The script will output a detailed log of the diagnostic process, including the consensus on the next best action.

```
2025-09-19 20:52:32,679 - agentic_diagnosis - INFO - {
  "action": {
    "test_name": "Chest X-ray",
    "reasoning": "Chest X-ray is the most important next test because it can definitively confirm or exclude pneumonia by showing characteristic lung infiltrates or consolidations. This will help discriminate the leading diagnosis of pneumonia from acute bronchitis or COPD exacerbation, which cannot be reliably distinguished based on clinical presentation alone. It is a cost-effective, widely recommended, and rapid diagnostic tool making it the optimal choice for this patient."
  },
  "consensus_summary": "The panel concurs that while pneumonia is the most likely diagnosis, the clinical overlap with acute bronchitis and COPD exacerbation warrants further diagnostic clarification. The chest X-ray was unanimously agreed upon as the priority test due to its high utility in confirming pneumonia presence or absence, which would guide further management decisions. This approach mitigates the risk of anchoring bias and ensures cost-effective, evidence-based patient care."
}
```

## Project Structure

```
pydantic-ai-exp/
├── README.md
├── main.py
├── pyproject.toml
├── templates/
│   ├── dr_challenger.jinja2
│   ├── dr_checklist.jinja2
│   ├── dr_decision_maker.jinja2
│   ├── dr_hypothesis.jinja2
│   ├── dr_stewardship.jinja2
│   └── dr_test_chooser.jinja2
└── utils/
    ├── log.py
    └── template_manager.py
```