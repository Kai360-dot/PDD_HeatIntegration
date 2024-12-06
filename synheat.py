""""This GAMSPy model is based upon the GAMS version originally published on
https://www.gams.com/latest/gamslib_ml/libhtml/
The design was originally published here:
Yee, T F, and Grossmann, I E, Simultaneous Optimization of Models for
Heat Integration - Heat Exchanger Network Synthesis. Computers and
Chemical Engineering 14, 10 (1990), 1151-1184.

For questions, remarks and corrections please reach out to:
contact@kairuth.ch
Written by: Kai Ruth 2024"""

import pandas as pd
import sys
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options
from gamspy.math import Min, Max, sqrt, log
import matplotlib.pyplot as plt
import json
import os
from settings import Settings
from tabulate import tabulate
from typing import Tuple

class SynheatModel:
    def __init__(self, minimum_heat_flux: bool, number_of_stages, name="super_heat", utilities_only = False):
        """Initializes the synheat model, note that the objective function is
        declared in the _build_model() method."""
        # Initialize container
        self.m = Container()
        self.settings = Settings()
        self.name = name
        self.num_hot_streams = self.settings.num_hot_streams
        self.num_cold_streams = self.settings.num_cold_streams
        self.num_stages = number_of_stages
        self.acc = self.settings.annual_capital_charge_ratio
        self.hours = self.settings.annual_operation_hours
        self.fh = self.settings.fcp_hot
        self.fc = self.settings.fcp_cold
        self.thin = self.settings.supply_temperature_hot_streams
        self.thout = self.settings.target_temperature_hot_streams
        self.tcin = self.settings.supply_temperature_cold_streams
        self.tcout = self.settings.target_temperature_cold_streams
        self.hh = self.settings.stream_film_coefficients_hot
        self.hc = self.settings.stream_film_coefficients_cold
        self.hucost = self.settings.heat_utility_cost
        self.cucost = self.settings.cooling_utility_cost
        self.unitc = self.settings.fixed_cost_heat_exchangers
        self.acoeff = self.settings.area_cost_coefficient_exchangers
        self.hucoeff = self.settings.area_cost_coefficient_heaters
        self.cucoeff = self.settings.area_cost_coefficient_coolers
        self.aexp = self.settings.cost_exponent_exchangers
        self.hhu = self.settings.film_coefficient_heating_utility
        self.hcu = self.settings.film_coefficient_cooling_utility
        self.thuin = self.settings.inlet_temperature_hot_utility
        self.thuout = self.settings.outlet_temperature_hot_utility
        self.tcuin = self.settings.inlet_temperature_cold_utility
        self.tcuout = self.settings.outlet_temperature_cold_utility
        self.tmapp = self.settings.minimum_exchange_temperature 
        self.bigM = self.settings.big_m_parameter
        self.minqhx = self.settings.minimum_heat_exchanged_in_heat_exchanger
        self.minimum_heat_flux = minimum_heat_flux
        self.utilities_only = utilities_only

        self.results = []
        self.tuples = []
        self.cut_sets = {}
        self.int_cuts = {}
        self.cut_counter = 0
        self._build_model()
        self.i, self.j, self.k, self.z = self.m['i'], self.m['j'], self.m['k'], self.m['z']

    def _build_model(self):
        """Builds the model using sets, parameters, variables and equations."""
        i = Set(
            container = self.m,
            name = "i",
            description = "hot",
            records = [f'{i}' for i in range(1, self.num_hot_streams+1)] 
        )
        j = Set(
            container = self.m,
            name = "j",
            description = "cold streams",
            records = [f'{i}' for i in range(1, self.num_cold_streams+1)] 
        )
        k = Set(
            container = self.m, 
            name = "k",
            description = "temperature locations nok + 1",
            records = [f'{i}' for i in range(1,self.num_stages + 2)]  
        )
        st = Set(
            container = self.m, 
            name = "st",
            description = "stages",
            domain=k,
            records = [f'{i}' for i in range(1,self.num_stages + 1)]  
        )
        first = Set(
            container = self.m, 
            name = "first",
            description = "first temperature location",
            domain= k,
            records = ['1']
        )
        last = Set(
            container = self.m, 
            name = "last",
            description = "last temperature location",
            records = [f'{self.num_stages + 1}']
        )

        # 2. Define Parameters
        acc = Parameter(
            container = self.m,  
            name = "acc",
            description = "annual capital charge",
            records = self.acc
        )
        hours = Parameter(
            container = self.m,  
            name = "hours",
            description = "operation time per year in hours",
            records = self.hours
        )
        fh = Parameter(
            container = self.m,  
            name = "fh",
            description = "heat capacity flowrate of hot stream",
            domain = i, 
            records = self.fh 
        )
        fc = Parameter(
            container = self.m,  
            name = "fc",
            description = "heat capacity flowrate of the cold stream",
            domain = j, 
            records = self.fc
        )
        thin = Parameter(
            container = self.m,  
            name = "thin",
            description = "supply temperature of hot stream",
            domain = i, 
            records = self.thin
        )
        thout = Parameter(
            container = self.m,  
            name = "thout",
            description = "target temperature of hot stream",
            domain = i, 
            records = self.thout
        )
        tcin = Parameter(
            container = self.m,  
            name = "tcin",
            description = "supply temperature of cold stream",
            domain = j, 
            records = self.tcin
        )
        tcout = Parameter(
            container = self.m,  
            name = "tcout",
            description = "target temperature of cold stream",
            domain = j, 
            records = self.tcout
        )
        ech = Parameter(
            container = self.m,  
            name = "ech",
            description = "heat content hot i [MW]",
            domain = i, 
        )
        ecc = Parameter(
            container = self.m,  
            name = "ecc",
            description = "heat content cold j [MW]",
            domain = j, 
        )
        hh = Parameter(
            container = self.m,  
            name = "hh",
            description = "stream individual film coefficient hot i",
            domain = i, 
            records = self.hh
        )
        hc = Parameter(
            container = self.m,  
            name = "hc",
            description = "stream-individual film coefficient cold j",
            domain = j, 
            records = self.hc
        )
        hucost = Parameter(
            container = self.m,  
            name = "hucost",
            description = "cost of heating utility [$ / MWh]",
            records = self.hucost 
        )
        cucost = Parameter(
            container = self.m,  
            name = "cucost",
            description = "cost of cooling utility [$ / MWh]",
            records = self.cucost
        )
        unitc = Parameter(
            container = self.m,  
            name = "unitc",
            description = "fixed charge for exchanger",
            records = self.unitc
        )
        acoeff = Parameter(
            container = self.m,  
            name = "acoeff",
            description = "area cost coefficient for exchangers",
            records = self.acoeff
        )
        hucoeff = Parameter(
            container = self.m,  
            name = "hucoeff",
            description = "area cost coefficient for heaters",
            records = self.hucoeff
        )
        cucoeff = Parameter(
            container = self.m,  
            name = "cucoeff",
            description = "area cost coefficient for coolers",
            records = self.cucoeff
        )
        aexp = Parameter(
            container = self.m,  
            name = "aexp",
            description = "cost exponent for exchangers",
            records = self.aexp
        )
        hhu = Parameter(
            container = self.m,  
            name = "hhu",
            description = "stream individual film coefficient hot utility",
            records = self.hhu 
        )
        hcu = Parameter(
            container = self.m,  
            name = "hcu",
            description = "stream individual film coefficient cold utility",
            records = self.hcu
        )
        thuin = Parameter(
            container = self.m,  
            name = "thuin",
            description = "inlet temperature hot utility",
            records = self.thuin
        )
        thuout = Parameter(
            container = self.m,  
            name = "thuout",
            description = "outlet temperature hot utility",
            records = self.thuout 
        )
        tcuin = Parameter(
            container = self.m,  
            name = "tcuin",
            description = "inlet temperature cold utility",
            records = self.tcuin 
        )
        tcuout = Parameter(
            container = self.m,  
            name = "tcuout",
            description = "outlet temperature cold utility",
            records = self.tcuout
        )
        gamma = Parameter(
            container = self.m,  
            name = "gamma",
            description = "upper bound of driving force",
            domain = [i, j], 
        )
        a = Parameter(
            container = self.m,  
            name = "a",
            description = "area for exchanger for match ij in interval k (chen approximation)",
            domain = [i, j, k], 
        )
        al = Parameter(
            container = self.m,  
            name = "al",
            description = "area calculated with log mean",
            domain = [i, j, k], 
        )
        acu = Parameter(
            container = self.m,  
            name = "acu",
            description = "area coolers",
            domain = i, 
        )
        ahu = Parameter(
            container = self.m,  
            name = "ahu",
            description = "area heaters",
            domain = j, 
        )
        tmapp = Parameter(
            container = self.m,  
            name = "tmapp",
            description = "minimum approach temperature",
            records = self.tmapp
        )
        costheat = Parameter(
            container = self.m,  
            name = "costheat",
            description = "cost of heating",
        )
        costcool = Parameter(
            container = self.m,  
            name = "costcool",
            description = "cost of cooling",
        )
        invcost = Parameter(
            container = self.m,  
            name = "invcost",
            description = "investment cost",
        )
        heat = Parameter(
            container = self.m,  
            name = "heat",
            description = "total amount of heat from utilities",
        )
        cool = Parameter(
            container = self.m,  
            name = "cool",
            description = "total amount of heat abstracted by cooling utilities",
        )
        bigM = Parameter(
            container = self.m,  
            name = "bigM",
            description = "big M parameter",
            records = self.bigM
        )
        minqhx = Parameter(
            container = self.m,  
            name = "minqhx",
            description = "minimum heat flow in a single heat exchanger, to avoid many heat exchangers with small flows [MW]",
            records = self.minqhx 
        )
        # 3. Define Variables
        #  ------------ Binary variables --------------------------
        z = Variable(
            container = self.m, 
            name = "z",
            domain = [i, j, k],  
            type = "Binary", 
            description = "Existence of heat exchanger between streams i and j at stage k"
        )
        zcu = Variable(
            container = self.m, 
            name = "zcu",
            domain = i,  
            type = "Binary", 
            description = "Existence of heat exchanger between cold utility and hot stream i"
        )
        zhu = Variable(
            container = self.m, 
            name = "zhu",
            domain = j,  
            type = "Binary", 
            description = "Existence of heat exchanger between hot utility and cold stream j"
        )
        #  ------------ Positive variables --------------------------
        th = Variable(
            container = self.m, 
            name = "th",
            domain = [i, k],  
            type = "Positive", 
            description = "temperature of hot stream i as it enters stage k"
        )
        tc = Variable(
            container = self.m,
            name = "tc",
            domain = [j, k],  
            type = "Positive", 
            description = "temperature of cold stream j as it leaves stage k"
        )
        q = Variable(
            container = self.m, 
            name = "q",
            domain = [i, j, k],  
            type = "Positive", 
            description = "heat exchanged between i and j in stage k"
        )
        qc = Variable(
            container = self.m, 
            name = "qc",
            domain = i,  
            type = "Positive", 
            description = "heat exchanged between i and the cold utility"
        )
        qh = Variable(
            container = self.m, 
            name = "qh",
            domain = j,  
            type = "Positive", 
            description = "heat exchanged between j and the hot utility"
        )
        dt = Variable(
            container = self.m, 
            name = "dt",
            domain = [i, j, k],  
            type = "Positive", 
            description = "approach temperature difference between i and j at location k"
        )
        dtcu = Variable(
            container = self.m, 
            name = "dtcu",
            domain = i,  
            type = "Positive", 
            description = "approach temperature difference between i and the cold utility"
        )
        dthu = Variable(
            container = self.m, 
            name = "dthu",
            domain = j,  
            type = "Positive", 
            description = "approach temperature difference between j and the hot utility"
        )
        #  ------------ General variables --------------------------
        cost = Variable(
            container = self.m, 
            name = "cost",
            description = "HEN and utility cost"
        )

        # 4. Define Constraints (Equations)
        eh = Equation(
            container = self.m,  
            name = "eh",
            description = "heat exchanged by i in stage k",
            domain = [i, k], 
        )
        eh[i, k].where[st[k]] = fh[i]*(th[i,k] - th[i,k.lead(1)]) == Sum(j, q[i,j,k]) 

        eqc = Equation(
            container = self.m,  
            name = "eqc",
            description = "heat exchanged by i with the cold utility",
            domain = [i, k], 
        )
        eqc[i, k].where[last[k]] = fh[i]*(th[i,k] - thout[i]) == qc[i] 

        teh = Equation(
            container = self.m,  
            name = "teh",
            description = "total heat exchanged by hot stream i",
            domain = i, 
            definition = (thin[i]-thout[i])*fh[i] == Sum([j,st], q[i,j,st]) + qc[i]  
        )
        ec = Equation(
            container = self.m,  
            name = "ec",
            description = "heat exchanged by cold stream j in stage k",
            domain = [j, k] 
        )
        ec[j, k].where[st[k]] = fc[j]*(tc[j,k] - tc[j,k.lead(1)]) == Sum(i, q[i,j,k]) 

        eqh = Equation(
            container = self.m,  
            name = "eqh",
            description = "heat exchanged by cold stream j with hot utility",
            domain = [j, k], 
        )
        eqh[j, k].where[first[k]] = fc[j]*(tcout[j] - tc[j,k]) == qh[j] 

        tec = Equation(
            container = self.m,  
            name = "tec",
            description = "total heat exchanged by cold stream j",
            domain = j, 
            definition = (tcout[j]-tcin[j])*fc[j] == Sum([i,st], q[i,j,st]) + qh[j]  
        )

        month = Equation(
            container = self.m,  
            name = "month",
            description = "monotonicity of th (temperature of hot stream upon entry to stage k)",
            domain = [i, k], 
        )
        month[i, k].where[st[k]] =  th[i,k] >= th[i,k.lead(1)] 

        montc = Equation(
            container = self.m,  
            name = "montc",
            description = "monotonicity of tc (temperature of cold stream j upon leaving stage k)",
            domain = [j, k], 
        )
        montc[j,k].where[st[k]] = tc[j,k] >= tc[j,k.lead(1)] ;

        monthl = Equation(
            container = self.m,  
            name = "monthl",
            description = "monotonicity of th k = last",
            domain = [i, k], 
        )
        monthl[i,k].where[last[k]] = th[i,k] >= thout[i] 

        montcf = Equation(
            container = self.m,  
            name = "montcf",
            description = "monotonicity of tc k = first",
            domain = [j, k], 
        )
        montcf[j,k].where[first[k]] = tcout[j] >= tc[j,k]

        tinh = Equation(
            container = self.m,  
            name = "tinh",
            description = "supply temperature of hot streams",
            domain = [i, k], 
        )
        tinh[i,k].where[first[k]] = thin[i] == th[i,k]

        tinc = Equation(
            container = self.m,  
            name = "tinc",
            description = "supply temperature of cold streams",
            domain = [j, k], 
        )
        tinc[j,k].where[last[k]] = tcin[j] == tc[j,k] 

        logq = Equation(
            container = self.m,  
            name = "logq",
            description = "logical constraints on q",
            domain = [i, j, k], 
        )
        logq[i,j,k].where[st[k]] = q[i,j,k] - Min(ech[i], ecc[j])*z[i,j,k] <= 0 

        logqh = Equation(
            container = self.m,  
            name = "logqh",
            description = "logical constraints on qh(j)",
            domain = j, 
            definition = qh[j] - ecc[j]*zhu[j] <= 0 
        )

        logqc = Equation(
            container = self.m,  
            name = "logqc",
            description = "logical constraints on qc(i)",
            domain = i, 
            definition = qc[i] - ech[i]*zcu[i] <= 0 
        )

        logdth = Equation(
            container = self.m,  
            name = "logdth",
            description = "logical constraints on dt at the hot end",
            domain = [i, j, k], 
        )
        logdth[i,j,k].where[st[k]] =  dt[i,j,k] <= th[i,k] - tc[j,k] + gamma[i,j]*(1 - z[i,j,k]) 

        logdtc = Equation(
            container = self.m,  
            name = "logdtc",
            description = "logical constraints on dt at the cold end",
            domain = [i, j, k], 
        )
        logdtc[i,j,k].where[st[k]] = dt[i,j,k.lead(1)] <= th[i,k.lead(1)]-tc[j,k.lead(1)] + gamma[i,j]*(1 - z[i,j,k]) 

        logdtcu = Equation(
            container = self.m,  
            name = "logdtcu",
            description = "logical constraints on dtcu",
            domain = [i, k], 
        )
        logdtcu[i,k].where[last[k]] = dtcu[i] <= th[i,k] - tcuout 

        logdthu = Equation(
            container = self.m,  
            name = "logdthu",
            description = "logical constraints on dthu",
            domain = [j, k], 
        )
        logdthu[j,k].where[first[k]] = dthu[j] <= (thuout - tc[j,k]) 

        if self.minimum_heat_flux:
            minqf = Equation(
            container = self.m,  
            name = "minqf",
            description = "Custom Added Equation Task G",
            domain = [i, j, k], 
            definition = q[i, j, k] >= minqhx - bigM*(1-z[i, j, k]) 
        )
        
        if self.utilities_only:
            no_internal_heat_exchange = Equation(
                container = self.m, # use your container name
                name = "no_internal_heat_exchange",
                description = "Forbidds internal heat exchange maximizing utility use",
                domain = [i, j, k], # Enter the name of the set, where it should apply to each member of.
                definition = q[i, j, k] == 0 # Enter the equation with '==', '<=', '>='
            )

        # Define bounds
        dt.lo[i,j,k] = tmapp 
        dthu.lo[j] = tmapp 
        dtcu.lo[i] = tmapp 

        th.up[i,k] = thin[i] 
        th.lo[i,k] = thout[i] 
        tc.up[j,k] = tcout[j] 
        tc.lo[j,k] = tcin[j] 

        # Initialization (required for convergence)
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

        # Define objective cost function: obj
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

        utility_cost = (Sum(j,qh[j]*hucost) + Sum(i,qc[i]*cucost))*hours # $ / MWh * h/yr * MW

        self.obj = (
            (fixed_charge_HX +
            var_charge_HX +
            var_charge_heater + 
            var_charge_cooler) * acc # to be paid upfront (mult. by acc)
            + utility_cost 
        )

    def solve_model(self):
        """Initializes and solves the model"""
        model = Model(
            container=self.m,
            name=self.name,
            equations=self.m.getEquations(),
            problem="MINLP",
            sense=Sense.MIN,
            objective=self.obj, 
        )
        model.solve(
            solver="DICOPT",
            output=sys.stdout,
            options=Options(
                equation_listing_limit=100,
                variable_listing_limit=100,
            ),
        )
        self._current_objective_value = model.objective_value
        return model

    def add_integer_cut(self, tuples_with_level_one):
        """Adds an integer cut based on the provided tuples."""
        self.cut_counter += 1
        cut_name = f"set_{self.cut_counter}"
        self.cut_sets[self.cut_counter] = Set(
            container=self.m,
            name=cut_name,
            domain=[self.i, self.j, self.k],
            description=f"set of cut {self.cut_counter}",
        )

        # Add tuples to set
        for i_val, j_val, k_val in tuples_with_level_one:
            self.cut_sets[self.cut_counter][i_val, j_val, k_val] = True

        # Add integer cut equation
        self.int_cuts[self.cut_counter] = Equation(
            container=self.m,
            name=f"equation_{self.cut_counter}",
            description=f"Implementation of integer cut {self.cut_counter}",
            definition=(
                Sum([self.i, self.j, self.k], self.z[self.i, self.j, self.k].where[self.cut_sets[self.cut_counter][self.i, self.j, self.k]])
                - Sum([self.i, self.j, self.k], self.z[self.i, self.j, self.k].where[~self.cut_sets[self.cut_counter][self.i, self.j, self.k]])
                <= len(tuples_with_level_one) - 1
            ),
        )

    def run_with_integer_cuts(self, max_cuts=10):
        """Run the model iteratively with integer cuts."""
        for _ in range(max_cuts + 1):
            model = self.solve_model()
            self.results.append(round(model.objective_value, 4))
            z_current = self.m['z'].records
            current_tuples = z_current[z_current['level']==1][['i', 'j', 'k']].apply(tuple, axis=1).tolist()
            self.tuples.append(current_tuples)
            self.add_integer_cut(current_tuples)

    def save_results_to_json(self, file_path):
        """Saves the results to a JSON file"""
        data = {
            "results": self.results,
            "tuples": self.tuples,
            "cut_counter": self.cut_counter
        }
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
    def load_results_from_json(self, file_path):
        """Loads the results from a JSON file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                self.results = data.get("results", [])
                self.tuples = data.get("tuples", [])
                self.cut_counter = data.get("cut_counter", 0)

    def plot_results(self, save_path=None):
        """Plots the results."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.results)), self.results)
        plt.ylabel("Cost / $(yr)^{-1}$")
        plt.xlabel("Number of Integer Cuts")
        plt.title(f"Impact of Integer Cuts on Objective Cost (stages: {self.num_stages})")
        plt.grid(True)
        if save_path:
            plt.savefig(f'{save_path}_stages_{self.num_stages}', dpi=300)
        plt.show()

    def _calculate_areas(self):
        m = self.m
        a,al, i, j, k, st, q, hh, hc, dt = (m['a'], 
                                            m['al'],
                                            m['i'], 
                                            m['j'], 
                                            m['k'], 
                                            m['st'], 
                                            m['q'], 
                                            m['hh'], 
                                            m['hc'], 
                                            m['dt'])
        # Using Chen approximation
        a[i,j,k].where[st[k]] = ((q.l[i,j,k]*((1/hh[i])+(1/hc[j])))/
                                (dt.l[i,j,k]*dt.l[i,j,k.lead(1)]*
                                (dt.l[i,j,k]+dt.l[i,j,k.lead(1)])/2)**0.33333)
        
        # Using Log mean temperature seems wrong!
        # al[i,j,k].where[st[k]] = ((q.l[i,j,k]*((1/hh[i])+(1/hc[j])))/
        #                         (dt.l[i,j,k]*dt.l[i,j,k.lead(1)]*
        #                         (dt.l[i,j,k]+dt.l[i,j,k.lead(1)])/2)**0.33333)
        
        U_inv = ((1/hh[i])+(1/hc[j]))
        Delta_T_1 = dt.l[i,j,k]
        Delta_T_2 = dt.l[i,j,k.lead(1)]

        Delta_T_logmean = (Delta_T_1 - Delta_T_2) / log(1e-6 + Delta_T_1 / Delta_T_2)
        
        # Note: The following statement doesn't work in GAMS thus the 
        # conditional where operator is implemented on the left hand side yielding
        # the equivalent result to the below expression:
        # if Delta_T_1 == Delta_T_2:
        #     Delta_T_logmean = (q.l[i,j,k]) *U_inv / Delta_T_1

        
        al[i,j,k].where[st[k]] = ((q.l[i,j,k] * U_inv) / Delta_T_logmean)
        al[i,j,k].where[((dt.l[i,j,k.lead(1)] - dt.l[i,j,k])**2 <= 1e-4) & (st[k])] = ((q.l[i,j,k] * U_inv) / dt.l[i,j,k])


        ahu, qh, hhu, thuin, tcout, dthu, qc, acu, thout, tcuin, dtcu, hcu = (m['ahu'], 
                                                                              m['qh'], 
                                                                              m['hhu'], 
                                                                              m['thuin'],
                                                                              m['tcout'],
                                                                              m['dthu'],
                                                                              m['qc'],
                                                                              m['acu'],
                                                                              m['thout'], 
                                                                              m['tcuin'], 
                                                                              m['dtcu'], 
                                                                              m['hcu'])

        ahu[j] = (qh.l[j]*((1/hc[j]) + (1/hhu))/(((thuin-tcout[j])*dthu.l[j]*
                ((thuin-tcout[j]+dthu.l[j]))/2) + 1e-6)**0.33333) 

        acu[i] = ((qc.l[i]*((1/hh[i])+(1/hcu))/(((thout[i]-tcuin)*dtcu.l[i]*
                 (thout[i]-tcuin+dtcu.l[i])/2 + 1e-6)**0.33333)) )

    def _calculate_utilities_cost(self):
        """Calculates the utilties cost in [USD / yr]"""
        m = self.m
        costheat, j, qh, hucost, i, cucost, hours, qc = (m['costheat'],
                                                        m['j'], 
                                                        m['qh'], 
                                                        m['hucost'], 
                                                        m['i'], 
                                                        m['cucost'], 
                                                        m['hours'], 
                                                        m['qc'])
        # When assigning values to a scalar parameter in gampspy:
        # provide ellipsis literal [...]
        costheat[...] = Sum(j,qh.l[j]*hucost)*hours
        costcool = m['costcool']
        costcool[...] = Sum(i,qc.l[i]*cucost)*hours

    def _calculate_fixed_investment_cost(self):
        invcost= self.m['invcost']
        # Note the below equation ommits the hours, since the utility cost are annual already
        invcost[...] = (self._current_objective_value - (self.m['costheat'] + self.m['costcool']))/self.m['acc']

    def print_data(self, active_exchangers = True, 
                   heat_exchanged_internal = True,
                   temperatures_hot_streams = True,
                   temperatures_cold_streams = True,
                   heat_exchanger_areas = True, 
                   utilities = True, 
                   utilities_cost = True,
                   fixed_investment_cost = True,
                   objective_cost = True, 
                   toLatex = False):
        _is_active = self.m['z'].records[self.m['z'].records['level']==1][['i', 'j', 'k']].apply(tuple, axis=1).tolist()
        active_z = self.m['z'].records[self.m['z'].records['level']==1][['i', 'j', 'k', 'level']]

        print("==================== DATA =====================\n")
        print(f'Current Number of integer cuts: {len(self.results) - 1} (no cuts <==> -1 )')

        if active_exchangers:
            print('\n')
            print("z-values of active heat exchangers inside the HEN:")
            print(active_z)
            if toLatex:
                print(active_z.to_latex(index = False, float_format="%.2f"))
            print('\n')
        q_internal = self.m['q'].records
        active_q_internal = q_internal[q_internal.apply(lambda row: (row['i'], row['j'], row['k']) in _is_active, axis=1)][['i', 'j', 'k', 'level']]

        if heat_exchanged_internal:
            print('heat exchanged (q[i, j, k]) inside HEN / [MW]\n', active_q_internal.round(2))
            if toLatex:
                print('heat exchanged (q[i, j, k]) inside HEN / [MW]\n', active_q_internal.to_latex(index = False, float_format="%.2f"))
            print('\n')

        th = self.m['th'].records
        if temperatures_hot_streams:
            print('temperature hot streams i at temperature location k')
            print(th[['i', 'k', 'level']].round(2))
            if toLatex:
                print(th[['i', 'k', 'level']].to_latex(index = False, float_format="%.2f"))
            print('\n')
        

        tc = self.m['tc'].records
        if temperatures_cold_streams:
            print('temperature cold streams j at temperature location k')
            print(tc[['j', 'k', 'level']].round(2))
            if toLatex:
                print(tc[['j', 'k', 'level']].to_latex(index = False, float_format="%.2f"))
            print('\n')


        if heat_exchanger_areas:
            self._calculate_areas()
            # all values
            chen_area = self.m['a'].records
            log_area = self.m['al'].records
            # filter by activeness
            chen_area = chen_area[chen_area.apply(lambda row: (row['i'], row['j'], row['k']) in _is_active, axis=1)]
            log_area = log_area[log_area.apply(lambda row: (row['i'], row['j'], row['k']) in _is_active, axis=1)]
            # internal areas
            print('area calculated by Chens Approximation [m2]\n', chen_area[['i', 'j', 'k', 'value']].round(2))
            if toLatex:
                print('area calculated by Chens Approximation [m2]\n', chen_area[['i', 'j', 'k', 'value']].to_latex(index = False, float_format="%.2f"))

            print('\n')
            print('area calculated with logmean temperature difference [m2]\n', log_area[['i', 'j', 'k', 'value']].round(2))
            if toLatex:
                print('area calculated with logmean temperature difference [m2]\n', log_area[['i', 'j', 'k', 'value']].to_latex(index = False, float_format="%.2f"))

            print('\n')
            # utilities areas
            print('hot utility hx area  with cold stream j [m2]\n', self.m['ahu'].records[['j', 'value']].round(2))
            if toLatex:
                print('hot utility hx area  with cold stream j [m2]\n', self.m['ahu'].records[['j', 'value']].to_latex(index = False, float_format="%.2f"))

            print('\n')
            print('cold utility hx area with hot stream i [m2]\n', self.m['acu'].records[['i', 'value']].round(2))
            if toLatex:
                print('cold utility hx area with hot stream i [m2]\n', self.m['acu'].records[['i', 'value']].to_latex(index = False, float_format="%.2f"))

            print('\n')

        if utilities:
            print("hot utility heat [MW]")
            qh = self.m['qh'].records
            print("heat q [MW] to cold streams from hot utilities")
            print(qh[['j', 'level']].round(2))
            if toLatex:
                print(qh[['j', 'level']].to_latex(index = False, float_format="%.2f"))

            print('\n')

            print("cold utility heat [MW]")
            qc = self.m['qc'].records
            print("heat q [MW] from hot streams to cold utilities")
            print(qc[['i', 'level']].round(2))
            if toLatex:
                print(qc[['i', 'level']].to_latex(index = False, float_format="%.2f"))
                
            print('\n')


        # Cost calculations:
        self._calculate_utilities_cost()
        if utilities_cost:
            print('The utilities cost: [USD / yr]')
            print('Heating cost:', self.m['costheat'].records['value'][0].round(2))
            print('Cooling cost:', self.m['costcool'].records['value'][0].round(2))
            print('\n')

        self._calculate_fixed_investment_cost()
        if fixed_investment_cost:
            print('The fixed investment cost: [USD] (total paid on day one)')
            print(self.m['invcost'].records['value'][0].round(2))
            print('\n')
        
        if objective_cost:
            print("Objective Annual Cost [USD / yr]")
            print(round(self._current_objective_value, 2))
            print("\n")
        
        if toLatex: # prints table of costs
            heating_cost = self.m['costheat'].records['value'][0]
            cooling_cost = self.m['costcool'].records['value'][0]
            investment_cost = self.m['invcost'].records['value'][0]
            annual_investment_cost = investment_cost * self.acc
            print("\\hline")
            print("Category & Cost\\\\")
            print("\\hline")
            print(f"Heating / $\\SI{{}}{{\\dollar\\per\\year}}$ & {heating_cost:.2f} \\\\")
            print(f"Cooling / $\\SI{{}}{{\\dollar\\per\\year}}$ & {cooling_cost:.2f}\\\\")
            print(f"Investment / $\\SI{{}}{{\\dollar\\per\\year}}$ & {annual_investment_cost:.2f}\\\\")
            print(f"Investment (total) / $\\SI{{}}{{\\dollar}}$ & {investment_cost:.2f}\\\\")
            print(f"Objective Cost / $\\SI{{}}{{\\dollar\\per\\year}}$ & {self._current_objective_value:.2f}\\\\")
            print("\\hline")


    def return_best_result(self, print_output = False) -> Tuple[int, float, str]:
        """In order for the results to be available
        the code has to be run with the required amount of integer cuts
        or alternatively the results need to be fetched from the appropriate JSON 
        using SynheatModel.load_results_from_json(file_path).
        
        Returns:
        Tuple[Int ,float, str]: [int_cut_index, objective_val, LaTeX table best z configuration]
        """
        min_index = np.argmin(self.results)
        best_result = self.results[min_index]
        best_z_tuple = self.tuples[min_index]
        best_z_tuple_tab = tabulate(best_z_tuple, headers= ["i", "j", "k"], tablefmt="latex")
        if print_output:
            print(f'Best result found: {best_result} \nafter number of integer cuts {min_index}')
            print(f'At z-configuration (active heat exchangers) \n{best_z_tuple_tab}')
        return min_index, best_result, best_z_tuple_tab

    def return_worst_result(self, print_output = False) -> Tuple[int, float, str]:
        """In order for the results to be available
        the code has to be run with the required amount of integer cuts
        or alternatively the results need to be fetched from the appropriate JSON 
        using SynheatModel.load_results_from_json(file_path).
        
        Returns:
        Tuple[Int ,float, str]: [int_cut_index, objective_val, LaTeX table worst z configuration]
        """
        max_index = np.argmax(self.results)
        worst_result = self.results[max_index]
        worst_z_tuple = self.tuples[max_index]
        worst_z_tuple_tab = tabulate(worst_z_tuple, headers= ["i", "j", "k"], tablefmt="latex")
        if print_output:
            print(f'Worst (optimal) result found: {worst_result} \nafter number of integer cuts {max_index}')
            print(f'At z-configuration (active heat exchangers) \n{worst_z_tuple_tab}')
        return max_index, worst_result, worst_z_tuple_tab

    
        
