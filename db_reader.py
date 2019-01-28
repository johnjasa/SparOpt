import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# open database of previously saved cases
cr = CaseReader("cases.sql")

# get a list of cases that were recorded by the driver
driver_cases = cr.list_cases()
print driver_cases

case = cr.get_case(driver_cases[1])

objectives = case.get_objectives()
design_vars = case.get_design_vars()
constraints = case.get_constraints()

print(design_vars['D_tower_p'])
print(design_vars['wt_tower_p'])
print(constraints['total_tower_fatigue_damage.total_tower_fatigue_damage'])
print(objectives['parallel.cond0.total_cost'])
