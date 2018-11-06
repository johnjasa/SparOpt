import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# open database of previously saved cases
cr = CaseReader("test.db")

# get a list of cases that were recorded by the driver
driver_cases = cr.list_cases('driver')

case = cr.get_case(driver_cases[5])

objectives = case.get_objectives()
design_vars = case.get_design_vars()
constraints = case.get_constraints()

print(design_vars['rho_ball'])
print(objectives['CoG_ball'])