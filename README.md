# EcoHome Energy Advisor
![Static Badge](https://img.shields.io/badge/Python-3.10%2B%20-blue)  ![Static Badge](https://img.shields.io/badge/LangGraph-Latest%20-green) ![Static Badge](https://img.shields.io/badge/License-MIT%20-yellow)

An AI-powered agentic energy advisor that optimizes home energy usage for cost savings and reduced carbon footprint using weather forecasts, dynamic pricing, historical data, and retrieval-augmented generation (RAG).

## Project Overview

EcoHome is a personal prototype I built during my career break to explore agentic AI in a real-world sustainability use case. The Energy Advisor is an intelligent agent that helps homeowners with solar panels, electric vehicles (EVs), smart thermostats, and appliances make optimal decisions — such as when to charge the EV, run the dishwasher, or adjust HVAC settings — to minimize electricity costs and environmental impact.
The agent reasons step-by-step over multiple data sources, uses tools autonomously, and provides personalized, actionable recommendations with estimated savings.

### Key outcomes from simulations
  
  - Achieved **15–30% simulated cost reductions** depending on location and usage patterns
  - Reduced estimated carbon emissions by **~20%** through maximized solar utilization and off-peak shifting
  - Demonstrates end-to-end agentic workflow with planning, tool calling, shared state, and safe execution

### Key Features

- **Weather-aware solar forecasting**: Uses weather forecasts to predict solar generation
- **Dynamic electricity pricing integration**: Considers time-of-day electricity prices for cost optimization
- **Historical usage analysis**: Personalizes advice from past consumption patterns
- **RAG powered energy tips**: Retrieves best practices and relevant energy-saving tips
- **Multi-device Optimization**: Handles EVs, HVAC, appliances, and solar systems
- **Cost Calculations**: Provides specific savings estimates and ROI analysis

## Architecture

<img width="1285" height="781" alt="image" src="https://github.com/user-attachments/assets/f38348ea-9416-4cd6-92e9-1d5875358018" />


## Example Interactions

**User**: "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
**Agent Response (simulated)**:
  Tomorrow's solar peak is between 11 AM–3 PM with clear skies. Off-peak pricing starts at 10 PM. **Recommended**: Pre-cool the battery in the morning using excess solar (11 AM–   2 PM), then top off overnight after 10 PM. Estimated savings: $2.40 vs. immediate charging, with 65% solar utilization.

### Other examples :

  - "What thermostat temperature should I set Wednesday afternoon if prices spike?"
  - "Suggest three ways to reduce energy use based on my history."
  - "How much can I save by running the dishwasher during off-peak hours?"

## Tech stack :
  - **LangChain & LangGraph** : Agent framework, tool integration, and stateful orchestration
  - **OpenAI GPT models** : Reasoning and embeddings
  - **ChromaDB** : Vector store for RAG
  - **SQLAlchemy + SQLite** : Local storage for historical usage and solar data
  - **External APIs** : Weather forecasts and electricity pricing (configurable)

## Project Structure

```
ecohome/
├── agent.py                   # Main Energy Advisor agent (LangGraph workflow)
├── tools.py                   # Custom tools for weather, pricing, DB queries, RAG, calculator
├── models/
│   └── energy.py              # SQLAlchemy models
├── data/
│   └── documents/             # Text files for RAG (tips & best practices)
├── notebooks/
│   ├── 01_db_setup.ipynb
│   ├── 02_rag_setup.ipynb
│   ├── 03_agent_evaluation.ipynb
│   └── 04_agent_run.ipynb
├── requirements.txt
├── .env.example               # Template for API keys
└── README.md

```
## Setup & Quick start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ecohome.git
cd ecohome
```

### 2. Install dependencies

Create a `.env` file with your API keys:

```bash
pip install -r requirements.txt
```
### 3. Set API keys(.env file)

OPENAI_API_KEY=your_key
WEATHER_API_KEY=your_key  # e.g., OpenWeatherMap or similar

### 4. Run the Notebooks

Execute the notebooks in order:

1. **01_db_setup.ipynb** - Set up the database and populate with sample data
2. **02_rag_setup.ipynb** - Configure the RAG pipeline for energy tips
3. **03_agent_evaluation.ipynb** - Test and evaluate the agent
4. **04_agent_run.ipynb** - Run the agent with example scenarios

## Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## Future Enhancements (Ideas)

  - Integrate real smart-home APIs (e.g., Tesla, Nest)
  - Add multi-day planning with constraint optimization
  - Support carbon intensity forecasts by region
  - UI dashboard for usage visualization

## License
MIT License – feel free to fork, modify, or use as inspiration.
Built by Pravir Sinha | December 2025
