import numpy as np
from scipy.optimize import root

from openmdao.api import ImplicitComponent

class MoorTenZero(ImplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=1., units='m')
		self.add_input('water_depth', val=1., units='m')
		self.add_input('EA_moor', val=1., units='N')
		self.add_input('mass_dens_moor', val=1., units='kg/m')
		self.add_input('len_hor_moor', val=1., units='m')
		self.add_input('len_tot_moor', val=1., units='m')

		self.add_output('moor_tension_zero', val=1., units='N')
		self.add_output('eff_length_zero', val=1., units='m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		#Horizontal tension and effective mooring line length at zero offset

		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']

		l_eff = outputs['eff_length_zero']
		t_star = outputs['moor_tension_zero'] / (mu * 9.80665)
		
		residuals['moor_tension_zero'] = l_tot - l_eff - l_tot_hor + t_star * np.arcsinh(l_eff / t_star) + outputs['moor_tension_zero'] * l_eff / EA
		residuals['eff_length_zero'] = h - mu * 9.80665 * l_eff**2. / (2. * EA) - t_star * (np.sqrt(1. + (l_eff / t_star)**2.) - 1.)

	def solve_nonlinear(self, inputs, outputs):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]

		def fun(x):
			t_star = x[1] / (mu * 9.80665)
			return [l_tot - x[0] - l_tot_hor + t_star * np.arcsinh(x[0] / t_star) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star * (np.sqrt(1. + (x[0] / t_star)**2.) - 1.)]

		sol = root(fun, [600., 1.0e6])

		outputs['eff_length_zero'] = sol.x[0]
		outputs['moor_tension_zero'] = sol.x[1]

	def linearize(self, inputs, outputs, partials):
		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']

		l_eff = outputs['eff_length_zero']
		t_star = outputs['moor_tension_zero'] / (mu * 9.80665)

		partials['moor_tension_zero', 'z_moor'] = 0.
		partials['moor_tension_zero', 'water_depth'] = 0.
		partials['moor_tension_zero', 'EA_moor'] = -outputs['moor_tension_zero'] * l_eff / EA**2.
		partials['moor_tension_zero', 'mass_dens_moor'] = (np.arcsinh(l_eff / t_star) - l_eff / t_star * 1. / np.sqrt(1 + (l_eff / t_star)**2.)) * (-outputs['moor_tension_zero'] / (mu**2. * 9.80665))
		partials['moor_tension_zero', 'len_hor_moor'] = -1.
		partials['moor_tension_zero', 'len_tot_moor'] = 1.
		partials['moor_tension_zero', 'moor_tension_zero'] = (np.arcsinh(l_eff / t_star) - l_eff / t_star * 1. / np.sqrt(1 + (l_eff / t_star)**2.)) * 1. / (mu * 9.80665) + l_eff / EA
		partials['moor_tension_zero', 'eff_length_zero'] = -1. + 1. / np.sqrt(1 + (l_eff / t_star)**2.) + outputs['moor_tension_zero'] / EA

		partials['eff_length_zero', 'z_moor'] = 1.
		partials['eff_length_zero', 'water_depth'] = 1.
		partials['eff_length_zero', 'EA_moor'] = mu * 9.80665 * l_eff**2. / (2. * EA**2.)
		partials['eff_length_zero', 'mass_dens_moor'] = 9.80665 * l_eff**2. / (2. * EA) - ((np.sqrt(1. + (l_eff / t_star)**2.) - 1.) + t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * (-2. * l_eff**2. / t_star**3.)) * (-outputs['moor_tension_zero'] / (mu**2. * 9.80665))
		partials['eff_length_zero', 'len_hor_moor'] = 0.
		partials['eff_length_zero', 'len_tot_moor'] = 0.
		partials['eff_length_zero', 'moor_tension_zero'] = -((np.sqrt(1. + (l_eff / t_star)**2.) - 1.) + t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * (-2. * l_eff**2. / t_star**3.)) * 1. / (mu * 9.80665)
		partials['eff_length_zero', 'eff_length_zero'] = -mu * 9.80665 * l_eff / EA - t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * 2. * l_eff / t_star**2.