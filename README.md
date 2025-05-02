# PromptGPT

**Quantitative Evaluation of Prompt Variants for LLMs**

This repository implements an automated framework for comparing different prompt variants when calling Large Language Models on a coding assignment (CNF Satisfiability). It generates, tests, grades, and analyzes model responses over hundreds of iterations.

---

##  Repository Structure

```
├── solution.py         # LLM-generated `sat(clauses)` implementation
├── testsuite.py        # 100-case CNF satisfiability unit tests
├── runner.py           # Entrypoint for Docker-based grading
├── testKey.ipynb       # Notebook: experiment loop, API calls, grading, CSV export
├── results.csv         # Collected scores: Model, Variant, Iteration, Score
├── Dockerfile          # Defines the `sat-tester` container image
└── README.md           # Project overview & instructions
```

---

##  Prerequisites

- **Python 3.10+**
- **Docker** (to run untrusted LLM-generated code safely)
- **OpenAI API Key** (export as `OPENAI_API_KEY`)
- **R (4.0+)** with packages: `readr`, `dplyr`, `ggplot2`

---

##  Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/henryfinnila/PromptGPT.git
   cd PromptGPT
   ```

2. **Build the Docker test image**:

   ```bash
   docker build -t sat-tester .
   ```

3. **Install Python dependencies** (optional venv):

   ```bash
   pip install -r requirements.txt
   ```

---

##  Running Unit Tests Locally

To ensure your `sat(clauses)` function passes all 100 test cases:

```bash
python -m unittest testsuite.py
```

---

##  Running Experiments

1. **Configure your OpenAI key**:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Run the experiment notebook**:

   - Open `testKey.ipynb` in JupyterLab or VS Code
   - Execute cells in order. This will:
     - Send prompts to various models & variants
     - Grade responses in the `sat-tester` container
     - Append results to `results.csv`

---

##  Statistical Analysis

Use the following R code to perform:

- One-way and Two-way ANOVA of scores by `variant` and `model`
- Histograms and boxplots of score distributions

Example in R:

```r
library(readr)
results <- read_csv("results.csv")
library(dplyr)
library(ggplot2)

# One-way ANOVA
aov_variant <- aov(score ~ variant, data = results)
summary(aov_variant)

# Two-way ANOVA
aov_two <- aov(score ~ model * variant, data = results)
summary(aov_two)
```

---

##  Contributions

- Feel free to submit issues or pull requests
- Add new prompt variants in `testKey.ipynb` or `runner.py`
- Extend to other coding assignments

---
