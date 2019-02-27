import numpy as np
from scipy.optimize import root, fsolve

from openmdao.api import ImplicitComponent

class MoorTenSigSurge(ImplicitComponent):

	def setup(self):
		self.add_input('moor_tension_sig_surge_offset', val=0., units='N')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')
		self.add_input('mass_dens_moor', val=0., units='kg/m')

		self.add_output('moor_ten_sig_surge_tot', val=0., units='N')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		T_H = inputs['moor_tension_sig_surge_offset']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		T = outputs['moor_ten_sig_surge_tot']

		residuals['moor_ten_sig_surge_tot'] = T_H - EA * np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) + EA

	def solve_nonlinear(self, inputs, outputs):
		T_H = inputs['moor_tension_sig_surge_offset']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		def fun(x):
			return T_H - EA * np.sqrt((x[0] / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) + EA

		#sol = root(fun, 1.0e6, tol=1e-5)
		sol = fsolve(fun, 1.0e6, xtol=1e-5)

		outputs['moor_ten_sig_surge_tot'] = sol[0]

	def linearize(self, inputs, outputs, partials):
		T_H = inputs['moor_tension_sig_surge_offset']
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		T = outputs['moor_ten_sig_surge_tot']

		partials['moor_ten_sig_surge_tot', 'moor_tension_sig_surge_offset'] = 1.
		partials['moor_ten_sig_surge_tot', 'z_moor'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * mu * 9.80665 / EA
		partials['moor_ten_sig_surge_tot', 'water_depth'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * mu * 9.80665 / EA
		partials['moor_ten_sig_surge_tot', 'EA_moor'] = -np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) - EA * 0.5 / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * (-2. * (T / EA + 1.) * T / EA**2. + 2. * mu * 9.80665 * h / EA**2.) + 1.
		partials['moor_ten_sig_surge_tot', 'mass_dens_moor'] = 0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * 9.80665 * h / EA
		partials['moor_ten_sig_surge_tot', 'moor_ten_sig_surge_tot'] = -0.5 * EA / np.sqrt((T / EA + 1.)**2. - 2. * mu * 9.80665 * h / EA) * 2. * (T / EA + 1.) / EA