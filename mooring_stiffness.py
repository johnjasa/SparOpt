import numpy as np
import scipy.optimize as so

from openmdao.api import ExplicitComponent

class MooringStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('water_depth', val=0., units='m')
		self.add_input('z_moor', val=0., units='m')
		#self.add_input('offset', val=0., units='m')

		self.add_output('K_moor', val=0., units='N/m')
		self.add_output('M_moor', val=0., units='kg')

	def compute(self, inputs, outputs):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = 384243000.
		mu = 155.41
		l_tot_hor = 848.67# + offset
		l_tot = 902.2

		step = 0.001

		def fun(x):
			t_star = x[1] / (mu * 9.80665)
			return [l_tot - x[0] - l_tot_hor + t_star * np.arcsinh(x[0] / t_star) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star * (np.sqrt(1. + (x[0] / t_star)**2.) -1.)]

		def fun_fin_diff(x):
			t_star = x[1] / (mu * 9.80665)
			return [l_tot - x[0] - (l_tot_hor + step) + t_star * np.arcsinh(x[0] / t_star) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star * (np.sqrt(1. + (x[0] / t_star)**2.) -1.)]

		sol = so.root(fun, [600.0, 1.0e6])

		sol_fin_diff = so.root(fun_fin_diff, [sol.x[0], sol.x[1]])

		l_eff = sol.x[0]
		t_hor = sol.x[1]

		outputs['M_moor'] = 330000.#3. * l_eff * mu
		outputs['K_moor'] = 1.5 * (sol_fin_diff.x[1] - sol.x[1]) / step #1.5 due to 3 lines (sum of cos(angle)^2)