import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# open database of previously saved cases
cr = CaseReader("cases.sql")

# get a list of cases that were recorded by the driver
driver_cases = cr.list_cases('problem')
print driver_cases
"""
case = cr.get_case(driver_cases[0])

objectives = case.get_objectives()
design_vars = case.get_design_vars()
constraints = case.get_constraints()

print(design_vars['z_moor'])
print(objectives['M_moor'])
"""