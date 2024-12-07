"""Print the data on the best result that attains an objective value of 
cost = 89123.5408,
using
\hline
   i &   j &   k \\
\hline
   1 &   2 &   1 \\
   1 &   3 &   2 \\
   2 &   3 &   3 \\
   3 &   2 &   2 \\
   3 &   3 &   4 \\
\hline"""
from synheat import SynheatModel

synheat_best_overall = SynheatModel(number_of_stages=4, minimum_heat_flux=True)
synheat_best_overall.run_with_integer_cuts(61)
synheat_best_overall.print_data(toLatex=True)
synheat_best_overall.return_best_result(print_output=True)