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
		partials['moor_tension_zero', 'mass_dens_moor'] = 0.#
		partials['moor_tension_zero', 'len_hor_moor'] = -1.
		partials['moor_tension_zero', 'len_tot_moor'] = 1.
		partials['moor_tension_zero', 'moor_tension_zero'] = 0.#
		partials['moor_tension_zero', 'eff_length_zero'] = 0.#

		partials['eff_length_zero', 'z_moor'] = 1.
		partials['eff_length_zero', 'water_depth'] = 1.
		partials['eff_length_zero', 'EA_moor'] = mu * 9.80665 * l_eff**2. / (2. * EA**2.)
		partials['eff_length_zero', 'mass_dens_moor'] = 0.#
		partials['eff_length_zero', 'len_hor_moor'] = 0.
		partials['eff_length_zero', 'len_tot_moor'] = 0.
		partials['eff_length_zero', 'moor_tension_zero'] = 0.#
		partials['eff_length_zero', 'eff_length_zero'] = 0.#