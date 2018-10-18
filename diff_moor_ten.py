import numpy as np

from openmdao.api import ImplicitComponent

class DiffMoorTen(ImplicitComponent):

	def setup(self):
		self.add_input('water_depth', val=0., units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')
		self.add_input('mass_dens_moor', val=0., units='kg/m')
		self.add_input('len_hor_moor', val=0., units='m')
		self.add_input('len_tot_moor', val=0., units='m')
		self.add_input('eff_length_offset_ww', val=0., units='m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_lw', val=0., units='m')
		self.add_input('moor_tension_offset_lw', val=0., units='N')

		self.add_output('deff_length_ww_dx', val=0., units='m/m')
		self.add_output('dmoor_tension_ww_dx', val=0., units='N/m')
		self.add_output('deff_length_lw_dx', val=0., units='m/m')
		self.add_output('dmoor_tension_lw_dx', val=0., units='N/m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']

		l_eff_ww = inputs['eff_length_offset_ww']
		t_hor_ww = inputs['moor_tension_offset_ww']
		l_eff_lw = inputs['eff_length_offset_lw']
		t_hor_lw = inputs['moor_tension_offset_lw']

		dl_eff_ww_dx = outputs['deff_length_ww_dx']
		dt_hor_ww_dx = outputs['dmoor_tension_ww_dx']
		dl_eff_lw_dx = outputs['deff_length_ww_dx']
		dt_hor_lw_dx = outputs['dmoor_tension_ww_dx']
		
		residuals['dmoor_tension_ww_dx'] = -dl_eff_ww_dx - 1. + 1. / (mu * 9.80665) * dt_hor_ww_dx * np.arcsinh(l_eff_ww * mu * 9.80665 / t_hor_ww) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + dt_hor_ww_dx * l_eff_ww / EA + t_hor_ww / EA * dl_eff_ww_dx
		residuals['deff_length_ww_dx'] = mu * 9.80665 * l_eff_ww / EA * dl_eff_ww_dx - 1. / (mu * 9.80665) * dt_hor_ww_dx * (np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) - 1.) - t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx)

		residuals['dmoor_tension_lw_dx'] = -dl_eff_lw_dx + 1. + 1. / (mu * 9.80665) * dt_hor_lw_dx * np.arcsinh(l_eff_lw * mu * 9.80665 / t_hor_lw) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + dt_hor_lw_dx * l_eff_lw / EA + t_hor_lw / EA * dl_eff_lw_dx
		residuals['deff_length_lw_dx'] = mu * 9.80665 * l_eff_lw / EA * dl_eff_lw_dx - 1. / (mu * 9.80665) * dt_hor_lw_dx * (np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) - 1.) - t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx)