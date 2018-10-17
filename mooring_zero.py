import numpy as np

from openmdao.api import ImplicitComponent

class MooringZero(ImplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('moor_tension_zero', val=0., units='N')
		self.add_output('eff_length_zero', val=0., units='m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		#Horizontal tension and effective mooring line length at zero offset

		h = inputs['water_depth'] + inputs['z_moor']
		EA = 384243000.
		mu = 155.41
		l_tot_hor = 848.67
		l_tot = 902.2

		l_eff = outputs['eff_length_zero']
		t_star = outputs['moor_tension_zero'] / (mu * 9.80665)
		
		residuals['moor_tension_zero'] = l_tot - l_eff - l_tot_hor + t_star * np.arcsinh(l_eff / t_star) + outputs['moor_tension_zero'] * l_eff / EA
		residuals['eff_length_zero'] = h - mu * 9.80665 * l_eff**2. / (2. * EA) - t_star * (np.sqrt(1. + (l_eff / t_star)**2.) - 1.)