import numpy as np

from openmdao.api import ExplicitComponent

class MooringStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('water_depth', val=0., units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('eff_length_offset_ww', val=0., units='m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_lw', val=0., units='m')
		self.add_input('moor_tension_offset_lw', val=0., units='N')

		self.add_output('deff_length_ww_dx', val=0., units='m/m')
		self.add_output('dmoor_tension_ww_dx', val=0., units='N/m')
		self.add_output('deff_length_lw_dx', val=0., units='m/m')
		self.add_output('dmoor_tension_lw_dx', val=0., units='N/m')
		self.add_output('K_moor', val=0., units='N/m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = 384243000.
		mu = 155.41
		l_tot_hor = 848.67
		l_tot = 902.2

		l_eff_ww = inputs['eff_length_offset_ww']
		t_hor_ww = inputs['moor_tension_offset_ww']

		l_eff_lw = inputs['eff_length_offset_lw']
		t_hor_lw = inputs['moor_tension_offset_lw']

		dl_eff_ww_dx = outputs['deff_length_offset_ww_dx']
		dt_hor_ww_dx = outputs['dmoor_tension_offset_ww_dx']
		dl_eff_lw_dx = outputs['deff_length_offset_ww_dx']
		dt_hor_lw_dx = outputs['dmoor_tension_offset_ww_dx']
		
		residuals['dmoor_tension_ww_dx'] = -dl_eff_ww_dx - 1. + 1. / (mu * g) * dt_hor_ww_dx * np.arcsinh(l_eff * mu * g / t_hor) + t_hor / (mu * g) * 1. / np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) * (mu * g / t_hor * dl_eff_ww_dx - l_eff * mu * g / t_hor**2. * dt_hor_ww_dx) + dt_hor_ww_dx * l_eff / EA + t_hor / EA * dl_eff_ww_dx
		residuals['deff_length_ww_dx'] = mu * g * l_eff / EA * dl_eff_ww_dx - 1. / (mu * g) * dt_hor_ww_dx * (np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) - 1.) - t_hor / (mu * g) * 1. / np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) * (l_eff * mu * g / t_hor) * (mu * g / t_hor * dl_eff_ww_dx - l_eff * mu * g / t_hor**2. * dt_hor_ww_dx)

		residuals['dmoor_tension_lw_dx'] = -dl_eff_lw_dx + 1. + 1. / (mu * g) * dt_hor_lw_dx * np.arcsinh(l_eff * mu * g / t_hor) + t_hor / (mu * g) * 1. / np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) * (mu * g / t_hor * dl_eff_lw_dx - l_eff * mu * g / t_hor**2. * dt_hor_lw_dx) + dt_hor_lw_dx * l_eff / EA + t_hor / EA * dl_eff_lw_dx
		residuals['deff_length_lw_dx'] = mu * g * l_eff / EA * dl_eff_lw_dx - 1. / (mu * g) * dt_hor_lw_dx * (np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) - 1.) - t_hor / (mu * g) * 1. / np.sqrt(1. + ((l_eff * mu * g) / t_hor)**2.) * (l_eff * mu * g / t_hor) * (mu * g / t_hor * dl_eff_lw_dx - l_eff * mu * g / t_hor**2. * dt_hor_lw_dx)

		residuals['K_moor'] = outputs['K_moor'] - dt_hor_ww_dx - 0.5 * dt_hor_lw_dx