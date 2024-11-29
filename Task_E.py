# Synheat adaptation in python with additional integer cut functionality
import pandas as pd
import sys
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options
from gamspy.math import Min, Max
# Initialize Container
m = Container()

# 1. Define Sets
# Hint: type 'set' and use autocomplete
i = Set(
    container = m,# use your container name
    name = "i",
    description = "hot",
    records = ['1', '2', '3'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)
j = Set(
    container = m,# use your container name
    name = "j",
    description = "cold streams",
    records = ['1', '2', '3'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)
k = Set(
    container = m,# use your container name
    name = "k",
    description = "temperature locations nok + 1",
    records = [f'{i}' for i in range(1,11)] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)
st = Set(
    container = m,# use your container name
    name = "st",
    description = "stages",
    domain=k,
    records = [f'{i}' for i in range(1,10)] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)
first = Set(
    container = m,# use your container name
    name = "first",
    description = "first temperature location",
    domain= k,
    records = ['1'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)
last = Set(
    container = m,# use your container name
    name = "last",
    description = "last temperature location",
    records = ['10'] # Enter like: ["Chicken", "Beef", "Eggs", "Milk"]
)

# 2. Define Parameters
# Hint: type 'parameter' and use autocomplete
# Hint: Use print(var_name.records) to verify parameter data.
nok = Parameter(
    container = m, # use your container name
    name = "nok",
    description = "number of stages in superstructure",
    # domain = [], # Enter the name of the set, where it should apply to each member of.
    records = [9] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
acc = Parameter(
    container = m, # use your container name
    name = "acc",
    description = "annual capital charge",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 0.205 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hours = Parameter(
    container = m, # use your container name
    name = "hours",
    description = "operation time per year in hours",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 8000 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
fh = Parameter(
    container = m, # use your container name
    name = "fh",
    description = "heat capacity flowrate of hot stream",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 4], ['2', 5], ['3', 3]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
fc = Parameter(
    container = m, # use your container name
    name = "fc",
    description = "heat capacity flowrate of the cold stream",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 6], ['2', 5], ['3', 5]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
thin = Parameter(
    container = m, # use your container name
    name = "thin",
    description = "supply temperature of hot stream",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 550], ['2', 400], ['3', 450]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
thout = Parameter(
    container = m, # use your container name
    name = "thout",
    description = "target temperature of hot stream",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 400], ['2', 300], ['3', 350]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
tcin = Parameter(
    container = m, # use your container name
    name = "tcin",
    description = "supply temperature of cold stream",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 460], ['2', 370], ['3', 300]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
tcout = Parameter(
    container = m, # use your container name
    name = "tcout",
    description = "target temperature of cold stream",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 570], ['2', 460], ['3', 470]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
ech = Parameter(
    container = m, # use your container name
    name = "ech",
    description = "heat content hot i [MW]",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
ecc = Parameter(
    container = m, # use your container name
    name = "ecc",
    description = "heat content cold j [MW]",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hh = Parameter(
    container = m, # use your container name
    name = "hh",
    description = "stream individual film coefficient hot i",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 1], ['2', 1], ['3', 1]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hc = Parameter(
    container = m, # use your container name
    name = "hc",
    description = "stream-individual film coefficient cold j",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    records = [['1', 1], ['2', 1], ['3', 1]] # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hucost = Parameter(
    container = m, # use your container name
    name = "hucost",
    description = "cost of heating utility [$ / MWh]",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 0.01 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
cucost = Parameter(
    container = m, # use your container name
    name = "cucost",
    description = "cost of cooling utility [$ / MWh]",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 0.0025 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
unitc = Parameter(
    container = m, # use your container name
    name = "unitc",
    description = "fixed charge for exchanger",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 10_000 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
acoeff = Parameter(
    container = m, # use your container name
    name = "acoeff",
    description = "area cost coefficient for exchangers",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 800 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hucoeff = Parameter(
    container = m, # use your container name
    name = "hucoeff",
    description = "area cost coefficient for heaters",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 800 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
cucoeff = Parameter(
    container = m, # use your container name
    name = "cucoeff",
    description = "area cost coefficient for coolers",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 800 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
aexp = Parameter(
    container = m, # use your container name
    name = "aexp",
    description = "cost exponent for exchangers",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 0.8 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hhu = Parameter(
    container = m, # use your container name
    name = "hhu",
    description = "stream individual film coefficient hot utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 5 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
hcu = Parameter(
    container = m, # use your container name
    name = "hcu",
    description = "stream individual film coefficient cold utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 1.0 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
thuin = Parameter(
    container = m, # use your container name
    name = "thuin",
    description = "inlet temperature hot utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 600 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
thuout = Parameter(
    container = m, # use your container name
    name = "thuout",
    description = "outlet temperature hot utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 590 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
tcuin = Parameter(
    container = m, # use your container name
    name = "tcuin",
    description = "inlet temperature cold utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 290 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
tcuout = Parameter(
    container = m, # use your container name
    name = "tcuout",
    description = "outlet temperature cold utility",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 305 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
gamma = Parameter(
    container = m, # use your container name
    name = "gamma",
    description = "upper bound of driving force",
    domain = [i, j], # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
a = Parameter(
    container = m, # use your container name
    name = "a",
    description = "area for exchanger for match ij in interval k (chen approximation)",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
al = Parameter(
    container = m, # use your container name
    name = "al",
    description = "area calculated with log mean",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
acu = Parameter(
    container = m, # use your container name
    name = "acu",
    description = "area coolers",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
ahu = Parameter(
    container = m, # use your container name
    name = "ahu",
    description = "area heaters",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
tmapp = Parameter(
    container = m, # use your container name
    name = "tmapp",
    description = "minimum approach temperature",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 10 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
costheat = Parameter(
    container = m, # use your container name
    name = "costheat",
    description = "cost of heating",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
costcool = Parameter(
    container = m, # use your container name
    name = "costcool",
    description = "cost of cooling",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    # records = records #  Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
invcost = Parameter(
    container = m, # use your container name
    name = "invcost",
    description = "investment cost",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
heat = Parameter(
    container = m, # use your container name
    name = "heat",
    description = "total amount of heat from utilities",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
cool = Parameter(
    container = m, # use your container name
    name = "cool",
    description = "total amount of heat abstracted by cooling utilities",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    # records = records # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
bigM = Parameter(
    container = m, # use your container name
    name = "bigM",
    description = "big M parameter",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 1e12 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
minqhx = Parameter(
    container = m, # use your container name
    name = "minqhx",
    description = "minimum heat flow in a single heat exchanger, to avoid many heat exchangers with small flows [MW]",
    # domain = set_name, # Enter the name of the set, where it should apply to each member of.
    records = 5 # Enter like this: [["Yellow", "Giraffe", 4], ["Brown", "Bear", 10]]
)
# 3. Define Variables
# Hint: type 'variable' and use autocomplete
#  ------------ Binary variables --------------------------
z = Variable(
    container = m, #use your container name
    name = "z",
    domain = [i, j, k],  # Enter the name of the set, where it should apply to each member of.
    type = "Binary", # e.g. "Positive"
    description = "Existence of heat exchanger between streams i and j at stage k"
)
zcu = Variable(
    container = m, #use your container name
    name = "zcu",
    domain = i,  # Enter the name of the set, where it should apply to each member of.
    type = "Binary", # e.g. "Positive"
    description = "Existence of heat exchanger between cold utility and hot stream i"
)
zhu = Variable(
    container = m, #use your container name
    name = "zhu",
    domain = j,  # Enter the name of the set, where it should apply to each member of.
    type = "Binary", # e.g. "Positive"
    description = "Existence of heat exchanger between hot utility and cold stream j"
)
#  ------------ Positive variables --------------------------
th = Variable(
    container = m, #use your container name
    name = "th",
    domain = [i, k],  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "temperature of hot stream i as it enters stage k"
)
tc = Variable(
    container = m, #use your container name
    name = "tc",
    domain = [j, k],  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "temperature of cold stream j as it leaves stage k"
)
q = Variable(
    container = m, #use your container name
    name = "q",
    domain = [i, j, k],  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "heat exchanged between i and j in stage k"
)
qc = Variable(
    container = m, #use your container name
    name = "qc",
    domain = i,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "heat exchanged between i and the cold utility"
)
qh = Variable(
    container = m, #use your container name
    name = "qh",
    domain = j,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "heat exchanged between j and the hot utility"
)
dt = Variable(
    container = m, #use your container name
    name = "dt",
    domain = [i, j, k],  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "approach temperature difference between i and j at location k"
)
dtcu = Variable(
    container = m, #use your container name
    name = "dtcu",
    domain = i,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "approach temperature difference between i and the cold utility"
)
dthu = Variable(
    container = m, #use your container name
    name = "dthu",
    domain = j,  # Enter the name of the set, where it should apply to each member of.
    type = "Positive", # e.g. "Positive"
    description = "approach temperature difference between j and the hot utility"
)
#  ------------ General variables --------------------------
cost = Variable(
    container = m, #use your container name
    name = "cost",
    # domain = set_name,  # Enter the name of the set, where it should apply to each member of.
    # type = "type", # e.g. "Positive"
    description = "HEN and utility cost"
)

# 4. Define Constraints (Equations)
# Hint: type 'equation' and use autocomplete
eh = Equation(
    container = m, # use your container name
    name = "eh",
    description = "heat exchanged by i in stage k",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
)
eh[i, k].where[st[k]] = fh[i]*(th[i,k] - th[i,k.lead(1)]) == Sum(j, q[i,j,k]) 

eqc = Equation(
    container = m, # use your container name
    name = "eqc",
    description = "heat exchanged by i with the cold utility",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
)
eqc[i, k].where[last[k]] = fh[i]*(th[i,k] - thout[i]) == qc[i] # Enter the equation with '==', '<=', '>='

teh = Equation(
    container = m, # use your container name
    name = "teh",
    description = "total heat exchanged by hot stream i",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    definition = (thin[i]-thout[i])*fh[i] == Sum([j,st], q[i,j,st]) + qc[i]  # Enter the equation with '==', '<=', '>='
)
ec = Equation(
    container = m, # use your container name
    name = "ec",
    description = "heat exchanged by cold stream j in stage k",
    domain = [j, k] # Enter the name of the set, where it should apply to each member of.
)
ec[j, k].where[st[k]] = fc[j]*(tc[j,k] - tc[j,k.lead(1)]) == Sum(i, q[i,j,k]) # Enter the equation with '==', '<=', '>='

eqh = Equation(
    container = m, # use your container name
    name = "eqh",
    description = "heat exchanged by cold stream j with hot utility",
    domain = [j, k], # Enter the name of the set, where it should apply to each member of.
)
eqh[j, k].where[first[k]] = fc[j]*(tcout[j] - tc[j,k]) == qh[j] # Enter the equation with '==', '<=', '>='

tec = Equation(
    container = m, # use your container name
    name = "tec",
    description = "total heat exchanged by cold stream j",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    definition = (tcout[j]-tcin[j])*fc[j] == Sum([i,st], q[i,j,st]) + qh[j]  # Enter the equation with '==', '<=', '>='
)

month = Equation(
    container = m, # use your container name
    name = "month",
    description = "monotonicity of th (temperature of hot stream upon entry to stage k)",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
)
month[i, k].where[st[k]] =  th[i,k] >= th[i,k.lead(1)] # Enter the equation with '==', '<=', '>='

montc = Equation(
    container = m, # use your container name
    name = "montc",
    description = "monotonicity of tc (temperature of cold stream j upon leaving stage k)",
    domain = [j, k], # Enter the name of the set, where it should apply to each member of.
)
montc[j,k].where[st[k]] = tc[j,k] >= tc[j,k.lead(1)] ;

monthl = Equation(
    container = m, # use your container name
    name = "monthl",
    description = "monotonicity of th k = last",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
)
monthl[i,k].where[last[k]] = th[i,k] >= thout[i] 

montcf = Equation(
    container = m, # use your container name
    name = "montcf",
    description = "monotonicity of tc k = first",
    domain = [j, k], # Enter the name of the set, where it should apply to each member of.
)
montcf[j,k].where[first[k]] = tcout[j] >= tc[j,k]

tinh = Equation(
    container = m, # use your container name
    name = "tinh",
    description = "supply temperature of hot streams",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
)
tinh[i,k].where[first[k]] = thin[i] == th[i,k]

tinc = Equation(
    container = m, # use your container name
    name = "tinc",
    description = "supply temperature of cold streams",
    domain = [j, k], # Enter the name of the set, where it should apply to each member of.
)
tinc[j,k].where[last[k]] = tcin[j] == tc[j,k] 

logq = Equation(
    container = m, # use your container name
    name = "logq",
    description = "logical constraints on q",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
)
logq[i,j,k].where[st[k]] = q[i,j,k] - Min(ech[i], ecc[j])*z[i,j,k] <= 0 

logqh = Equation(
    container = m, # use your container name
    name = "logqh",
    description = "logical constraints on qh(j)",
    domain = j, # Enter the name of the set, where it should apply to each member of.
    definition = qh[j] - ecc[j]*zhu[j] <= 0 
 # Enter the equation with '==', '<=', '>='
)

logqc = Equation(
    container = m, # use your container name
    name = "logqc",
    description = "logical constraints on qc(i)",
    domain = i, # Enter the name of the set, where it should apply to each member of.
    definition = qc[i] - ech[i]*zcu[i] <= 0 # Enter the equation with '==', '<=', '>='
)

logdth = Equation(
    container = m, # use your container name
    name = "logdth",
    description = "logical constraints on dt at the hot end",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
)
logdth[i,j,k].where[st[k]] =  dt[i,j,k] <= th[i,k] - tc[j,k] + gamma[i,j]*(1 - z[i,j,k]) 

logdtc = Equation(
    container = m, # use your container name
    name = "logdtc",
    description = "logical constraints on dt at the cold end",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
    # definition = enter equation # Enter the equation with '==', '<=', '>='
)
logdtc[i,j,k].where[st[k]] = dt[i,j,k.lead(1)] <= th[i,k.lead(1)]-tc[j,k.lead(1)] + gamma[i,j]*(1 - z[i,j,k]) 

logdtcu = Equation(
    container = m, # use your container name
    name = "logdtcu",
    description = "logical constraints on dtcu",
    domain = [i, k], # Enter the name of the set, where it should apply to each member of.
    # definition = enter equation # Enter the equation with '==', '<=', '>='
)
logdtcu[i,k].where[last[k]] = dtcu[i] <= th[i,k] - tcuout 

logdthu = Equation(
    container = m, # use your container name
    name = "logdthu",
    description = "logical constraints on dthu",
    domain = [j, k], # Enter the name of the set, where it should apply to each member of.
    # definition = enter equation # Enter the equation with '==', '<=', '>='
)
logdthu[j,k].where[first[k]] = dthu[j] <= (thuout - tc[j,k]) 

minqf = Equation(
    container = m, # use your container name
    name = "minqf",
    description = "Custom Added Equation Task G",
    domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
    definition = q[i, j, k] >= minqhx - bigM*(1-z[i, j, k]) # Enter the equation with '==', '<=', '>='
)

# 5. Define Objective Function

fixed_charge_HX = unitc * (Sum([i,j,st],z[i,j,st]) + Sum(i,zcu[i]) + Sum(j,zhu[j])) 

U_inv_HX = ((1 / hh[i]) + (1 / hc[j]))
chen_approx_HX = ( (dt[i,j,k] * dt[i,j,k.lead(1)] * (dt[i,j,k] + dt[i,j,k.lead(1)])/2 + 1e-6 ) **0.33333 )
var_charge_HX = acoeff * Sum([i,j,k], (q[i,j,k] * U_inv_HX/(chen_approx_HX + 1e-6) + 1e-6)**aexp) 

U_inv_heater = ((1/hc[j])+1/hhu)
chen_approx_heater = ( ((thuin-tcout[j])*dthu[j] * ((thuin-tcout[j]+dthu[j]) /2) +1e-6)**0.33333 )
var_charge_heater = hucoeff*(Sum(j,(qh[j]*U_inv_heater)/chen_approx_heater + 1e-6)**aexp) 

U_inv_cooler = ((1/hh[i])+(1/hcu))
chen_approx_cooler = ( ((thout[i]-tcuin)*dtcu[i]*(thout[i]-tcuin+dtcu[i])/2 +1e-6) **0.33333 )
var_charge_cooler = cucoeff*Sum(i,(qc[i]*U_inv_cooler/chen_approx_cooler + 1e-6)**aexp) 

utility_cost = (Sum(j,qh[j]*hucost) + Sum(i,qc[i]*cucost))*hours

obj = (
    (fixed_charge_HX +
    var_charge_HX +
    var_charge_heater + 
    var_charge_cooler) 
    * acc # to be paid upfront (mult. by acc)
    + utility_cost 
)
               
# Bounds
dt.lo[i,j,k] = tmapp 
dthu.lo[j] = tmapp 
dtcu.lo[i] = tmapp 

th.up[i,k] = thin[i] 
th.lo[i,k] = thout[i] 
tc.up[j,k] = tcout[j] 
tc.lo[j,k] = tcin[j] 

# Initialization
th.l[i,k] = thin[i]
tc.l[j,k] = tcin[j] 

dthu.l[j] = thuout - tcin[j] 
dtcu.l[i] = thin[i] - tcuout 

ech[i] = fh[i]*(thin[i] - thout[i]) 
ecc[j] = fc[j]*(tcout[j] - tcin[j]) 


gamma[i,j] = Max(0,
                 tcin[j] - thin[i], 
                 tcin[j] - thout[i], 
                 tcout[j] - thin[i], 
                 tcout[j] - thout[i])

dt.l[i,j,k] = thin[i] - tcin[j]

q.l[i,j,k].where[st[k]] = Min(ech[i],ecc[j])

# ADD Integer cut loop 

# 6. Create and Solve the Model
# Hint: type 'model' and use autocomplete
super_heat = Model(
    container = m, # use your container name,
    name = "super_heat",
    equations = m.getEquations(),
    problem = "MINLP", # could be LP, NLP, MILP, MINLP
    sense = Sense.MIN, # Either: MIN or MAX
    objective = obj # Enter the var. name of objective function
)
# Hint: type 'solve' and use autocomplete
super_heat.solve(
    solver = "DICOPT",
    output = sys.stdout,
    options = Options(
        equation_listing_limit = 100,
        variable_listing_limit = 100
    )
)

# 7. Print Results
print(super_heat.getVariableListing())
print(f"Optimal Cost: {super_heat.objective_value}")

super_heat.toLatex(path="/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/TEX/Task_E", generate_pdf=False)

# Generate GAMS File using:
super_heat.toGams(path="/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/GAMS/Task_E/SuperHeat.gms")