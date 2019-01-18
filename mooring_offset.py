import numpy as np
from scipy.optimize import root

from openmdao.api import ImplicitComponent

class MooringOffset(ImplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('F0_tower_drag', val=0., units='N')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=1., units='N')
		self.add_input('mass_dens_moor', val=1., units='kg/m')
		self.add_input('len_hor_moor', val=0., units='m')
		self.add_input('len_tot_moor', val=0., units='m')

		self.add_output('moor_tension_offset_ww', val=1., units='N')
		self.add_output('eff_length_offset_ww', val=1., units='m')
		self.add_output('moor_tension_offset_lw', val=1., units='N')
		self.add_output('eff_length_offset_lw', val=1., units='m')
		self.add_output('moor_offset', val=0., units='m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]

		l_eff_ww = outputs['eff_length_offset_ww']
		t_star_ww = outputs['moor_tension_offset_ww'] / (mu * 9.80665)

		l_eff_lw = outputs['eff_length_offset_lw']
		t_star_lw = outputs['moor_tension_offset_lw'] / (mu * 9.80665)
		
		residuals['moor_tension_offset_ww'] = l_tot - l_eff_ww - (l_tot_hor + outputs['moor_offset']) + t_star_ww * np.arcsinh(l_eff_ww / t_star_ww) + outputs['moor_tension_offset_ww'] * l_eff_ww / EA
		residuals['eff_length_offset_ww'] = h - mu * 9.80665 * l_eff_ww**2. / (2. * EA) - t_star_ww * (np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.)

		residuals['moor_tension_offset_lw'] = l_tot - l_eff_lw - (l_tot_hor - outputs['moor_offset']) + t_star_lw * np.arcsinh(l_eff_lw / t_star_lw) + outputs['moor_tension_offset_lw'] * l_eff_lw / EA
		residuals['eff_length_offset_lw'] = h - mu * 9.80665 * l_eff_lw**2. / (2. * EA) - t_star_lw * (np.sqrt(1. + (l_eff_lw / t_star_lw)**2.) - 1.)

		residuals['moor_offset'] = outputs['moor_tension_offset_ww'] - outputs['moor_tension_offset_lw'] - inputs['thrust_0'] - inputs['F0_tower_drag']

	def solve_nonlinear(self, inputs, outputs):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]
		thrust_0 = inputs['thrust_0'][0]
		F0_tower_drag = inputs['F0_tower_drag'][0]

		def fun(x):
			t_star_ww = x[1] / (mu * 9.80665)
			t_star_lw = x[3] / (mu * 9.80665)
			return [l_tot - x[0] - l_tot_hor - x[4] + t_star_ww * np.arcsinh(x[0] / t_star_ww) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star_ww * (np.sqrt(1. + (x[0] / t_star_ww)**2.) - 1.), l_tot - x[2] - l_tot_hor + x[4] + t_star_lw * np.arcsinh(x[2] / t_star_lw) + x[3] * x[2] / EA, h - mu * 9.80665 * x[2]**2. / (2. * EA) - t_star_lw * (np.sqrt(1. + (x[2] / t_star_lw)**2.) -1.), x[1] - x[3] - thrust_0 - F0_tower_drag]

		sol = root(fun, [600.0, 1.0e6, 600.0, 1.0e6, 1.]) #TODO: zero tension and eff. length as inital guess?

		outputs['eff_length_offset_ww'] = sol.x[0]
		outputs['moor_tension_offset_ww'] = sol.x[1]
		outputs['eff_length_offset_lw'] = sol.x[2]
		outputs['moor_tension_offset_lw'] = sol.x[3]
		outputs['moor_offset'] = sol.x[4]


	def linearize(self, inputs, outputs, partials):
		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']

		l_eff_ww = outputs['eff_length_offset_ww']
		t_star_ww = outputs['moor_tension_offset_ww'] / (mu * 9.80665)

		l_eff_lw = outputs['eff_length_offset_lw']
		t_star_lw = outputs['moor_tension_offset_lw'] / (mu * 9.80665)

		partials['moor_tension_offset_ww', 'thrust_0'] = 0.
		partials['moor_tension_offset_ww', 'z_moor'] = 0.
		partials['moor_tension_offset_ww', 'water_depth'] = 0.
		partials['moor_tension_offset_ww', 'EA_moor'] = -outputs['moor_tension_offset_ww'] * l_eff_ww / EA**2.
		partials['moor_tension_offset_ww', 'mass_dens_moor'] = (np.arcsinh(l_eff_ww / t_star_ww) - l_eff_ww / t_star_ww * 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.)) * (-outputs['moor_tension_offset_ww'] / (mu**2. * 9.80665))
		partials['moor_tension_offset_ww', 'len_hor_moor'] = -1.
		partials['moor_tension_offset_ww', 'len_tot_moor'] = 1.
		partials['moor_tension_offset_ww', 'moor_tension_offset_ww'] = (np.arcsinh(l_eff_ww / t_star_ww) - l_eff_ww / t_star_ww * 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.)) * 1. / (mu * 9.80665) + l_eff_ww / EA
		partials['moor_tension_offset_ww', 'eff_length_offset_ww'] = -1. + 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.) + outputs['moor_tension_offset_ww'] / EA
		partials['moor_tension_offset_ww', 'moor_tension_offset_lw'] = 0.
		partials['moor_tension_offset_ww', 'eff_length_offset_lw'] = 0.
		partials['moor_tension_offset_ww', 'moor_offset'] = -1.

		partials['eff_length_offset_ww', 'thrust_0'] = 0.
		partials['eff_length_offset_ww', 'z_moor'] = 1.
		partials['eff_length_offset_ww', 'water_depth'] = 1.
		partials['eff_length_offset_ww', 'EA_moor'] = mu * 9.80665 * l_eff_ww**2. / (2. * EA**2.)
		partials['eff_length_offset_ww', 'mass_dens_moor'] = 9.80665 * l_eff_ww**2. / (2. * EA) - ((np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.) + t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * (-2. * l_eff_ww**2. / t_star_ww**3.)) * (-outputs['moor_tension_offset_ww'] / (mu**2. * 9.80665))
		partials['eff_length_offset_ww', 'len_hor_moor'] = 0.
		partials['eff_length_offset_ww', 'len_tot_moor'] = 0.
		partials['eff_length_offset_ww', 'moor_tension_offset_ww'] = -((np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.) + t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * (-2. * l_eff_ww**2. / t_star_ww**3.)) * 1. / (mu * 9.80665)
		partials['eff_length_offset_ww', 'eff_length_offset_ww'] = -mu * 9.80665 * l_eff_ww / EA - t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * 2. * l_eff_ww / t_star_ww**2.
		partials['eff_length_offset_ww', 'moor_tension_offset_lw'] = 0.
		partials['eff_length_offset_ww', 'eff_length_offset_lw'] = 0.
		partials['eff_length_offset_ww', 'moor_offset'] = 0.

		partials['moor_tension_offset_lw', 'thrust_0'] = 0.
		partials['moor_tension_offset_lw', 'z_moor'] = 0.
		partials['moor_tension_offset_lw', 'water_depth'] = 0.
		partials['moor_tension_offset_lw', 'EA_moor'] = -outputs['moor_tension_offset_lw'] * l_eff_lw / EA**2.
		partials['moor_tension_offset_lw', 'mass_dens_moor'] = (np.arcsinh(l_eff_lw / t_star_lw) - l_eff_lw / t_star_lw * 1. / np.sqrt(1 + (l_eff_lw / t_star_lw)**2.)) * (-outputs['moor_tension_offset_lw'] / (mu**2. * 9.80665))
		partials['moor_tension_offset_lw', 'len_hor_moor'] = -1.
		partials['moor_tension_offset_lw', 'len_tot_moor'] = 1.
		partials['moor_tension_offset_lw', 'moor_tension_offset_ww'] = 0.
		partials['moor_tension_offset_lw', 'eff_length_offset_ww'] = 0.
		partials['moor_tension_offset_lw', 'moor_tension_offset_lw'] = (np.arcsinh(l_eff_lw / t_star_lw) - l_eff_lw / t_star_lw * 1. / np.sqrt(1 + (l_eff_lw / t_star_lw)**2.)) * 1. / (mu * 9.80665) + l_eff_lw / EA
		partials['moor_tension_offset_lw', 'eff_length_offset_lw'] = -1. + 1. / np.sqrt(1 + (l_eff_lw / t_star_lw)**2.) + outputs['moor_tension_offset_lw'] / EA
		partials['moor_tension_offset_lw', 'moor_offset'] = 1.

		partials['eff_length_offset_lw', 'thrust_0'] = 0.
		partials['eff_length_offset_lw', 'z_moor'] = 1.
		partials['eff_length_offset_lw', 'water_depth'] = 1.
		partials['eff_length_offset_lw', 'EA_moor'] = mu * 9.80665 * l_eff_lw**2. / (2. * EA**2.)
		partials['eff_length_offset_lw', 'mass_dens_moor'] = 9.80665 * l_eff_lw**2. / (2. * EA) - ((np.sqrt(1. + (l_eff_lw / t_star_lw)**2.) - 1.) + t_star_lw * (0.5 / np.sqrt(1. + (l_eff_lw / t_star_lw)**2.)) * (-2. * l_eff_lw**2. / t_star_lw**3.)) * (-outputs['moor_tension_offset_lw'] / (mu**2. * 9.80665))
		partials['eff_length_offset_lw', 'len_hor_moor'] = 0.
		partials['eff_length_offset_lw', 'len_tot_moor'] = 0.
		partials['eff_length_offset_lw', 'moor_tension_offset_ww'] = 0.
		partials['eff_length_offset_lw', 'eff_length_offset_ww'] = 0.
		partials['eff_length_offset_lw', 'moor_tension_offset_lw'] = -((np.sqrt(1. + (l_eff_lw / t_star_lw)**2.) - 1.) + t_star_lw * (0.5 / np.sqrt(1. + (l_eff_lw / t_star_lw)**2.)) * (-2. * l_eff_lw**2. / t_star_lw**3.)) * 1. / (mu * 9.80665)
		partials['eff_length_offset_lw', 'eff_length_offset_lw'] = -mu * 9.80665 * l_eff_lw / EA - t_star_lw * (0.5 / np.sqrt(1. + (l_eff_lw / t_star_lw)**2.)) * 2. * l_eff_lw / t_star_lw**2.
		partials['eff_length_offset_lw', 'moor_offset'] = 0.

		partials['moor_offset', 'thrust_0'] = -1.
		partials['moor_offset', 'F0_tower_drag'] = -1.
		partials['moor_offset', 'z_moor'] = 0.
		partials['moor_offset', 'water_depth'] = 0.
		partials['moor_offset', 'EA_moor'] = 0.
		partials['moor_offset', 'mass_dens_moor'] = 0.
		partials['moor_offset', 'len_hor_moor'] = 0.
		partials['moor_offset', 'len_tot_moor'] = 0.
		partials['moor_offset', 'moor_tension_offset_ww'] = 1.
		partials['moor_offset', 'eff_length_offset_ww'] = 0.
		partials['moor_offset', 'moor_tension_offset_lw'] = -1.
		partials['moor_offset', 'eff_length_offset_lw'] = 0.
		partials['moor_offset', 'moor_offset'] = 0.