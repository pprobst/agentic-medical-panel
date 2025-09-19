# Ageintic Medical Diagnosis Simulation

This project implements a multi-agent system using [`pydantic-ai`][https://ai.pydantic.dev/] to simulate a medical diagnostic process. It is heavily inspired by the concepts outlined in Microsoft's research on AI-driven diagnostics: [The Path to Medical Superintelligence](https://microsoft.ai/new/the-path-to-medical-superintelligence/).

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
2025-07-21 23:50:24,560 - agentic_diagnosis - INFO - --- Orchestrator Conclusion ---
2025-07-21 23:50:24,560 - agentic_diagnosis - INFO - {
  "action": {
    "test_name": "Chest X-ray",
    "reasoning": "Chest X-ray is the most critical next step as it can confirm or rule out community-acquired pneumonia by identifying typical infiltrates or consolidation. This will help to decisively differentiate between the leading hypothesis of pneumonia and other possibilities like acute bronchitis or COVID-19, which have different radiographic features."
  },
  "consensus_summary": "The leading diagnosis is community-acquired pneumonia based on the symptoms and patient age, but certainty is below 95%. The panel advises against sputum culture due to low yield and approves chest X-ray and COVID-19 PCR for diagnostic clarification. The debate panel highlighted possible anchoring bias on pneumonia diagnosis and recommended chest X-ray as a decisive test to confirm or refute pneumonia. COVID-19 PCR test is also approved for exclusion given symptom overlap and public health importance. The consensus is that chest X-ray is the critical next diagnostic action to guide management."
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