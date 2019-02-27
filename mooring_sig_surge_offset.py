import numpy as np
from scipy.optimize import root, fsolve

from openmdao.api import ImplicitComponent

class MooringSigSurgeOffset(ImplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=1., units='N')
		self.add_input('mass_dens_moor', val=1., units='kg/m')
		self.add_input('len_hor_moor', val=0., units='m')
		self.add_input('len_tot_moor', val=0., units='m')
		self.add_input('moor_offset', val=0., units='m')
		self.add_input('stddev_surge_WF', val=0., units='m')

		self.add_output('moor_tension_sig_surge_offset', val=1., units='N')
		self.add_output('eff_length_sig_surge_offset', val=1., units='m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]
		moor_offset = inputs['moor_offset'][0]
		stddev_surge_WF = inputs['stddev_surge_WF'][0]

		l_eff = outputs['eff_length_sig_surge_offset']
		t_star = outputs['moor_tension_sig_surge_offset'] / (mu * 9.80665)
		
		residuals['moor_tension_sig_surge_offset'] = l_tot - l_eff - (l_tot_hor + inputs['moor_offset'] + 2. * inputs['stddev_surge_WF']) + t_star * np.arcsinh(l_eff / t_star) + outputs['moor_tension_sig_surge_offset'] * l_eff / EA
		residuals['eff_length_sig_surge_offset'] = h - mu * 9.80665 * l_eff**2. / (2. * EA) - t_star * (np.sqrt(1. + (l_eff / t_star)**2.) - 1.)

	def solve_nonlinear(self, inputs, outputs):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]
		moor_offset = inputs['moor_offset'][0]
		stddev_surge_WF = inputs['stddev_surge_WF'][0]

		def fun(x):
			t_star = x[1] / (mu * 9.80665)
			return [l_tot - x[0] - l_tot_hor - moor_offset - 2. * stddev_surge_WF + t_star * np.arcsinh(x[0] / t_star) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star * (np.sqrt(1. + (x[0] / t_star)**2.) - 1.)]

		sol = fsolve(fun, [600.0, 1.0e6], xtol=1e-5)

		outputs['eff_length_sig_surge_offset'] = sol[0]
		outputs['moor_tension_sig_surge_offset'] = sol[1]

	def linearize(self, inputs, outputs, partials): #TODO check
		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']
		moor_offset = inputs['moor_offset']
		stddev_surge_WF = inputs['stddev_surge_WF']

		l_eff = outputs['eff_length_sig_surge_offset']
		t_star = outputs['moor_tension_sig_surge_offset'] / (mu * 9.80665)

		partials['moor_tension_sig_surge_offset', 'z_moor'] = 0.
		partials['moor_tension_sig_surge_offset', 'water_depth'] = 0.
		partials['moor_tension_sig_surge_offset', 'EA_moor'] = -outputs['moor_tension_sig_surge_offset'] * l_eff / EA**2.
		partials['moor_tension_sig_surge_offset', 'mass_dens_moor'] = (np.arcsinh(l_eff / t_star) - l_eff / t_star * 1. / np.sqrt(1 + (l_eff / t_star)**2.)) * (-outputs['moor_tension_sig_surge_offset'] / (mu**2. * 9.80665))
		partials['moor_tension_sig_surge_offset', 'len_hor_moor'] = -1.
		partials['moor_tension_sig_surge_offset', 'len_tot_moor'] = 1.
		partials['moor_tension_sig_surge_offset', 'moor_tension_sig_surge_offset'] = (np.arcsinh(l_eff / t_star) - l_eff / t_star * 1. / np.sqrt(1 + (l_eff / t_star)**2.)) * 1. / (mu * 9.80665) + l_eff / EA
		partials['moor_tension_sig_surge_offset', 'eff_length_sig_surge_offset'] = -1. + 1. / np.sqrt(1 + (l_eff / t_star)**2.) + outputs['moor_tension_sig_surge_offset'] / EA
		partials['moor_tension_sig_surge_offset', 'moor_offset'] = -1.
		partials['moor_tension_sig_surge_offset', 'stddev_surge_WF'] = -2.

		partials['eff_length_sig_surge_offset', 'z_moor'] = 1.
		partials['eff_length_sig_surge_offset', 'water_depth'] = 1.
		partials['eff_length_sig_surge_offset', 'EA_moor'] = mu * 9.80665 * l_eff**2. / (2. * EA**2.)
		partials['eff_length_sig_surge_offset', 'mass_dens_moor'] = -9.80665 * l_eff**2. / (2. * EA) - ((np.sqrt(1. + (l_eff / t_star)**2.) - 1.) + t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * (-2. * l_eff**2. / t_star**3.)) * (-outputs['moor_tension_sig_surge_offset'] / (mu**2. * 9.80665))
		partials['eff_length_sig_surge_offset', 'len_hor_moor'] = 0.
		partials['eff_length_sig_surge_offset', 'len_tot_moor'] = 0.
		partials['eff_length_sig_surge_offset', 'moor_tension_sig_surge_offset'] = -((np.sqrt(1. + (l_eff / t_star)**2.) - 1.) + t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * (-2. * l_eff**2. / t_star**3.)) * 1. / (mu * 9.80665)
		partials['eff_length_sig_surge_offset', 'eff_length_sig_surge_offset'] = -mu * 9.80665 * l_eff / EA - t_star * (0.5 / np.sqrt(1. + (l_eff / t_star)**2.)) * 2. * l_eff / t_star**2.