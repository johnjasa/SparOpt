import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# open database of previously saved cases
cr = CaseReader("integrated.sql")

# get a list of cases that were recorded by the driver
driver_cases = cr.list_cases()
print driver_cases

case = cr.get_case(driver_cases[-1])

objectives = case.get_objectives()
design_vars = case.get_design_vars()
constraints = case.get_constraints()

print(design_vars['D_spar_cp'])
print(design_vars['L_spar_cp'])
print(design_vars['D_tower_cp'])
print(design_vars['wt_tower_cp'])
print(design_vars['z_moor'])
print(design_vars['k_p'])
print(design_vars['k_i'])
print(design_vars['D_moor'])
print(design_vars['len_hor_moor'])
print(design_vars['len_tot_moor'])
print(constraints['parallel_ext.cond0_ext.substructure.T_heave'])
print(objectives['parallel_ext.cond0_ext.total_cost'])
