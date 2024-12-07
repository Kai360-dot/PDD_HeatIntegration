# Transshipment model
import pandas as pd
import sys
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options

# Initialize Container
cont = Container()

# 1. Define Sets
m = Set(
    container = cont,
    name = "m",
    description = "set of hot utilities (only steam here)",
    records = ['1'] 
)

n = Set(
    container = cont,
    name = "n",
    description = "set of cold utilities (only cooling water here)",
    records = ['1'] 
)

k = Set(
    container = cont,
    name = "k",
    description = "set of intervals",
    records = [f'{i}' for i in range(1, 10)] 
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
    container = cont, 
    name = "ch",
    description = "unit cost of hot utility m [$/MWh]",
    domain = m, 
    records = [['1', 0.01]] 
)
cc = Parameter(
    container = cont, 
    name = "cc",
    description = "unit cost of cold utility n [$/MWh]",
    domain = n, 
    records = [['1', 0.0025]] 
)

deficit = Parameter(
    container = cont, 
    name = "def",
    description = "deficit on interval k",
    domain = k, 
    records = [['1', -180], 
               ['2', -140], 
               ['3', -70], 
               ['4', -120], 
               ['5', -150], 
               ['6', -40], 
               ['7', 90], 
               ['8', 0], 
               ['9', 50], 
               ] 
)

# 3. Define Variables
r = Variable(
    container = cont, 
    name = "r",
    domain = k,  
    type = "Positive", 
    description = "residual at stage k"
)

qs = Variable(
    container = cont, 
    name = "qs",
    domain = m,  
    type = "Positive", 
    description = "heat load of utility m (given hot utility)"
)

qw = Variable(
    container = cont, 
    name = "qw",
    domain = n,  
    type = "Positive", 
    description = "heat load of utility n (given cold utility)"
)


# 4. Define Constraints (Equations)
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
    container = cont, 
    name = "MinUtil",
    equations = cont.getEquations(),
    problem = "LP", 
    sense = Sense.MIN, 
    objective = z 
)
MinUtil.solve(
    output = sys.stdout,
    options = Options(
        equation_listing_limit = 50,
        variable_listing_limit = 50
    )
)
# 7. Print Results
print('Residuals: \n',r.records)
print('Cooling:\n', qw.records)
print('Heating:\n', qs.records)
print(f"Optimal Cost: {MinUtil.objective_value}")
k_list = r.records['k'].tolist()
residual_list = r.records['level'].tolist()
for i, k in enumerate(k_list):
        print(f'{k} & {residual_list[i]} \\\\')