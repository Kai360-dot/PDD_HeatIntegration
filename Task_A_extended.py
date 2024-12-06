"""Computes the total cost of maximized utilities use
now including the heat exchanger costs."""

from synheat import SynheatModel

synheat_max_util = SynheatModel(minimum_heat_flux=False, number_of_stages=2, utilities_only=True)
synheat_max_util.solve_model()
synheat_max_util.print_data(toLatex=False)