# Transshipment model
import pandas as pd
import sys
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options

# Initialize Container
cont = Container()

# 1. Define Sets
m = Set(
    container = cont,# use your container name
    name = "m",
    description = "set of hot utilities (only steam here)",
    records = ['1'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)

n = Set(
    container = cont,# use your container name
    name = "n",
    description = "set of cold utilities (only cooling water here)",
    records = ['1'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)

k = Set(
    container = cont,# use your container name
    name = "k",
    description = "set of intervals",
    records = [f'{i}' for i in range(1, 10)] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)

k_int = Set(
    container=cont,
    name="k_int",
    domain = k, # k_int is a subset of k
    description="Intermediate intervals (2 to 8)",
    records=[f'{i}' for i in range(2, 9)]  # Explicitly define the intermediate intervals
)

# 2. Define Parameters
ch = Parameter(
    container = cont, # use your container name
    name = "ch",
    description = "unit cost of hot utility m [$/MWh]",
    domain = m, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 0.01]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
cc = Parameter(
    container = cont, # use your container name
    name = "cc",
    description = "unit cost of cold utility n [$/MWh]",
    domain = n, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 0.0025]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)

# 3. Define Variables
r = Variable(
    container = cont, #use your container name
    name = "r",
    domain = k,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "residual at stage k"
)

qs = Variable(
    container = cont, #use your container name
    name = "qs",
    domain = m,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "heat load of utility m (given hot utility)"
)

qw = Variable(
    container = cont, #use your container name
    name = "qw",
    domain = n,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "heat load of utility n (given cold utility)"
)

# 4. Define Constraints (Equations)
deficit = Parameter(
    container = cont, # use your container name
    name = "def",
    description = "deficit on interval k",
    domain = k, # Enter the name of the set, where it should apply to each member of.
    records = [['1', -180], 
               ['2', -140], 
               ['3', -70], 
               ['4', -120], 
               ['5', -450], 
               ['6', -40], 
               ['7', 90], 
               ['8', 0], 
               ['9', 50], 
               ] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)

r_int_eq = Equation(
    container = cont,
    name = "r_int_eq",
    description = "Heat transfer relation for intermediate intervals (2 to 8)",
    domain = k,  # Exclude first and last intervals
    definition = (r[k]).where[k_int[k]] - (r[k.lag(1)]).where[k_int[k]] == (deficit[k]).where[k_int[k]]
)

r_start_eq = Equation(
    container = cont,
    name = "r_start_eq",
    description = "Heat transfer relation for k=1",
    definition = r['1'] - qs['1'] == deficit['1']
)
r_end_eq = Equation(
    container = cont,
    name = "r_end_eq",
    description = "Heat transfer relation for k=9",
    definition = r['9'] - r['8'] + qw['1'] == deficit['9']
)

r.fx['9'] = 0

# 5. Define Objective Function
z = Sum(m, ch[m]*qs[m]) + Sum(n, cc[n]*qw[n])

# 6. Create and Solve the Model
MinUtil = Model(
    container = cont, # use your container name,
    name = "MinUtil",
    equations = cont.getEquations(),
    problem = "LP", # could be LP, NLP, MILP, MINLP
    sense = Sense.MIN, # Either: MIN or MAX
    objective = z # Enter the var. name of objective function
)
MinUtil.solve(
    output = sys.stdout,
    options = Options(
        equation_listing_limit = 50,
        variable_listing_limit = 50
    )
)
# 7. Print Results
# print(MinUtil.getVariableListing())
print('Residuals: \n',r.records)
print('Cooling:\n', qw.records)
print('Heating:\n', qs.records)
print(f"Optimal Cost: {MinUtil.objective_value}")
# print('k_int_records', k_int.records)
MinUtil.toLatex(path="/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/TEX/Task_C", generate_pdf=False)
# Generate GAMS File using:
MinUtil.toGams(path="/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/GAMS/Task_C")