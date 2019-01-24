import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# open database of previously saved cases
cr = CaseReader("cases.sql")

# get a list of cases that were recorded by the driver
driver_cases = cr.list_cases()
print driver_cases

case = cr.get_case(driver_cases[-2])

objectives = case.get_objectives()
design_vars = case.get_design_vars()
constraints = case.get_constraints()

print(design_vars['D_tower_p'])
print(design_vars['wt_tower_p'])
print(constraints['parallel.cond0.substructure.buoy_mass'])
print(objectives['parallel.cond0.total_cost'])
