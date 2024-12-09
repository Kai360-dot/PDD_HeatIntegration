# Heat Integration Model from the PDD Class at ETHZ

This repository contains a Python-based implementation of a heat integration model. The model is derived from the work originally published in:

**Yee, T. F., and Grossmann, I. E.,**  
*Simultaneous Optimization of Models for Heat Integration - Heat Exchanger Network Synthesis.*  
Computers and Chemical Engineering, 14(10), 1151-1184, 1990.

The code in this repository is adapted from the example code (`synheat`) available in the GAMS model library:  
[https://www.gams.com/latest/gamslib_ml/libhtml/](https://www.gams.com/latest/gamslib_ml/libhtml/)

### Key Features

- **Original GAMS Code Compatibility**: The base model was originally written in GAMS for mathematical programming.
- **Python Integration**: This implementation uses the `GAMSPy` Python package, interacting with the GAMS framework through a container-based structure.
- **Flexibility**: Easily modify the problem’s mathematical structure and run multiple scenarios in loops.
- **Enhanced Data Handling**: Store, manipulate, and visualize resulting data seamlessly using Python libraries.

---

## Getting Started

Modify the `settings.py` file as needed. Below is a complete example of how to use the `SynheatModel` class, from instantiation to data saving and plotting.

```python
# Import the SynheatModel class
from synheat import SynheatModel

# Initialize the model
your_model_name = SynheatModel(minimum_heat_flux=False, number_of_stages=4)

# Run the model without integer cuts
your_model_name.run_with_integer_cuts(max_cuts=0)

# Run the model with integer cuts (e.g., 5 cuts)
your_model_name.run_with_integer_cuts(5)

# Print the resulting data
your_model_name.print_data(toLatex=False)

# Retrieve the best result
index_of_best_value, best_result_objective_value, best_z_tuple_table = your_model_name.return_best_result(print_output=True)

# Retrieve the worst result
index_of_worst_value, worst_result_objective_value, worst_z_tuple_table = your_model_name.return_worst_result(print_output=True)

# Save results to a JSON file
your_model_name.save_results_to_json("ENTER YOUR FILEPATH")

# Load results from a JSON file (if file doesn't exist, run with specified cuts and save results)
your_model_name.load_results_from_json("ENTER YOUR FILEPATH")
if not your_model_name.results:
    your_model_name.run_with_integer_cuts(max_cuts=100)
    your_model_name.save_results_to_json("ENTER YOUR FILEPATH")

# Plot the obtained results (optional: save the plot to a file)
your_model_name.plot_results(save_path="ENTER YOUR SAVE PATH")
