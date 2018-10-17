import numpy as np

from openmdao.api import ImplicitComponent

class MooringOffset(ImplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('moor_tension_zero', val=0., units='N')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('moor_tension_offset_ww', val=0., units='N')
		self.add_output('eff_length_offset_ww', val=0., units='m')
		self.add_output('moor_tension_offset_lw', val=0., units='N')
		self.add_output('eff_length_offset_lw', val=0., units='m')
		self.add_output('moor_offset', val=0., units='m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'] + inputs['z_moor']
		EA = 384243000.
		mu = 155.41
		l_tot_hor = 848.67
		l_tot = 902.2

		l_eff_ww = outputs['eff_length_offset_ww']
		t_star_ww = outputs['moor_tension_offset_ww'] / (mu * 9.80665)

		l_eff_lw = outputs['eff_length_offset_lw']
		t_star_lw = outputs['moor_tension_offset_lw'] / (mu * 9.80665)
		
		residuals['moor_tension_offset_ww'] = l_tot - l_eff_ww - (l_tot_hor + outputs['moor_offset']) + t_star_ww * np.arcsinh(l_eff_ww / t_star_ww) + outputs['moor_tension_offset_ww'] * l_eff_ww / EA
		residuals['eff_length_offset_ww'] = h - mu * 9.80665 * l_eff_ww**2. / (2. * EA) - t_star_ww * (np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.)

		residuals['moor_tension_offset_lw'] = l_tot - l_eff_lw - (l_tot_hor - outputs['moor_offset']) + t_star_lw * np.arcsinh(l_eff_lw / t_star_lw) + outputs['moor_tension_offset_lw'] * l_eff_lw / EA
		residuals['eff_length_offset_lw'] = h - mu * 9.80665 * l_eff_lw**2. / (2. * EA) - t_star_lw * (np.sqrt(1. + (l_eff_lw / t_star_lw)**2.) - 1.)

		residuals['moor_offset'] = outputs['moor_tension_offset_ww'] + 0.5 * t_hor_lw - 1.5 * inputs['moor_tension_zero'] - inputs['thrust_0']