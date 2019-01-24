import numpy as np
from scipy.optimize import root

from openmdao.api import ImplicitComponent

class MeanMoorTen(ImplicitComponent):

	def setup(self):
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')
		self.add_input('mass_dens_moor', val=0., units='kg/m')

		self.add_output('mean_moor_ten', val=0., units='N')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		T_H = inputs['moor_tension_offset_ww']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		T = outputs['mean_moor_ten']

		residuals['mean_moor_ten'] = T_H - EA * np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) + EA

	def solve_nonlinear(self, inputs, outputs):
		T_H = inputs['moor_tension_offset_ww']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		def fun(x):
			return T_H - EA * np.sqrt((x[0] / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) + EA

		sol = root(fun, 1.0e6)

		outputs['mean_moor_ten'] = sol.x[0]

	def linearize(self, inputs, outputs, partials):
		T_H = inputs['moor_tension_offset_ww']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		T = outputs['mean_moor_ten']

		partials['mean_moor_ten', 'moor_tension_offset_ww'] = 1.
		partials['mean_moor_ten', 'z_moor'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * mu * 9.80665 / EA
		partials['mean_moor_ten', 'water_depth'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * mu * 9.80665 / EA
		partials['mean_moor_ten', 'EA_moor'] = -np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) - EA * 0.5 / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * (-2. * (T / EA + 1.) * T / EA**2. + 2. * mu * 9.80665 * h / EA**2.) + 1.
		partials['mean_moor_ten', 'mass_dens_moor'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * 9.80665 * h / EA
		partials['mean_moor_ten', 'mean_moor_ten'] = -0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * (T / EA + 1.) / EA