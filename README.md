# SherpAI-Schemas

Shared schema definitions for the SherpAI data quality pipeline.

## Installation

```bash
pip install git+https://github.com/SherpAI-Master/SherpAI-Schemas.git
```

## Schemas

### ProblemInstance
Tracks identified data quality issues in a row, categorized by type: `incomplete`, `misplaced`, `formatting`, `misspelled`, `missing_value`, `validation`.

### SolutionInstance
Holds proposed fixes for each column of a row. Each fix is a `Fix` object containing a `value` and a `reason`.

### Fix
Base fix object with an optional `value` and `reason`.

### MetaDataInstance
Preserves process events and execution order metadata.

## Usage

```python
from sherpai_schemas import ProblemInstance, Fix, SolutionInstance

# Create a problem instance
problem = ProblemInstance(incomplete=["col1"], misspelled=["col2"])
print(problem)  # 1['col1']4['col2']

# Create a solution
solution = SolutionInstance()
solution.name1 = Fix(value="John", reason="Corrected spelling")

# Parse from string
problem = ProblemInstance.parse_from_str("1['col1']4['col2']")
solution = SolutionInstance.parse_from_str("name1['John','Corrected spelling']")
```

## Requirements

- Python >= 3.12
- pandas >= 3.0.1

## Authors

Roman Klinghammer (rklinghammer@uni-potsdam.de)
