import numpy as np

from openmdao.api import ImplicitComponent

class MoorTenZero(ImplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')
		self.add_input('mass_dens_moor', val=0., units='kg/m')
		self.add_input('len_hor_moor', val=0., units='m')
		self.add_input('len_tot_moor', val=0., units='m')

		self.add_output('moor_tension_zero', val=0., units='N')
		self.add_output('eff_length_zero', val=0., units='m')

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