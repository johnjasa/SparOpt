import numpy as np
from scipy.optimize import root, fsolve

from openmdao.api import ImplicitComponent

class MaxMooringOffset(ImplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=0., units='m')
		self.add_input('water_depth', val=0., units='m')
		self.add_input('EA_moor', val=1., units='N')
		self.add_input('mass_dens_moor', val=1., units='kg/m')
		self.add_input('len_hor_moor', val=0., units='m')
		self.add_input('len_tot_moor', val=0., units='m')

		self.add_output('moor_tension_max_offset_ww', val=1., units='N')
		self.add_output('eff_length_max_offset_ww', val=1., units='m')
		self.add_output('maxval_fairlead', val=0., units='m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]

		l_eff_ww = outputs['eff_length_max_offset_ww']
		t_star_ww = outputs['moor_tension_max_offset_ww'] / (mu * 9.80665)
		
		residuals['moor_tension_max_offset_ww'] = l_tot - l_eff_ww - (l_tot_hor + outputs['maxval_fairlead']) + t_star_ww * np.arcsinh(l_eff_ww / t_star_ww) + outputs['moor_tension_max_offset_ww'] * l_eff_ww / EA
		residuals['eff_length_max_offset_ww'] = h - mu * 9.80665 * l_eff_ww**2. / (2. * EA) - t_star_ww * (np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.)

		residuals['maxval_fairlead'] = outputs['moor_tension_max_offset_ww'] / (mu * 9.80665) * np.arcsinh(mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww']) + outputs['moor_tension_max_offset_ww'] * outputs['eff_length_max_offset_ww'] / EA - l_tot_hor

	def solve_nonlinear(self, inputs, outputs):
		h = inputs['water_depth'][0] + inputs['z_moor'][0]
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]
		l_tot_hor = inputs['len_hor_moor'][0]
		l_tot = inputs['len_tot_moor'][0]

		def fun(x):
			t_star_ww = x[1] / (mu * 9.80665)
			return [l_tot - x[0] - l_tot_hor - x[2] + t_star_ww * np.arcsinh(x[0] / t_star_ww) + x[1] * x[0] / EA, h - mu * 9.80665 * x[0]**2. / (2. * EA) - t_star_ww * (np.sqrt(1. + (x[0] / t_star_ww)**2.) - 1.), x[1] / (mu * 9.80665) * np.arcsinh(mu * 9.80665 * x[0] / x[1]) + x[1] * x[0] / EA - l_tot_hor]

		#sol = root(fun, [600.0, 1.0e6, 5.], method='krylov', tol=1e-5)
		sol = fsolve(fun, [600.0, 1.0e6, 10.], xtol=1e-5)

		outputs['eff_length_max_offset_ww'] = sol[0]
		outputs['moor_tension_max_offset_ww'] = sol[1]
		outputs['maxval_fairlead'] = sol[2]

	def linearize(self, inputs, outputs, partials):
		h = inputs['water_depth'] + inputs['z_moor']
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']
		l_tot_hor = inputs['len_hor_moor']
		l_tot = inputs['len_tot_moor']

		l_eff_ww = outputs['eff_length_max_offset_ww']
		t_star_ww = outputs['moor_tension_max_offset_ww'] / (mu * 9.80665)

		partials['moor_tension_max_offset_ww', 'z_moor'] = 0.
		partials['moor_tension_max_offset_ww', 'water_depth'] = 0.
		partials['moor_tension_max_offset_ww', 'EA_moor'] = -outputs['moor_tension_max_offset_ww'] * l_eff_ww / EA**2.
		partials['moor_tension_max_offset_ww', 'mass_dens_moor'] = (np.arcsinh(l_eff_ww / t_star_ww) - l_eff_ww / t_star_ww * 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.)) * (-outputs['moor_tension_max_offset_ww'] / (mu**2. * 9.80665))
		partials['moor_tension_max_offset_ww', 'len_hor_moor'] = -1.
		partials['moor_tension_max_offset_ww', 'len_tot_moor'] = 1.
		partials['moor_tension_max_offset_ww', 'moor_tension_max_offset_ww'] = (np.arcsinh(l_eff_ww / t_star_ww) - l_eff_ww / t_star_ww * 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.)) * 1. / (mu * 9.80665) + l_eff_ww / EA
		partials['moor_tension_max_offset_ww', 'eff_length_max_offset_ww'] = -1. + 1. / np.sqrt(1 + (l_eff_ww / t_star_ww)**2.) + outputs['moor_tension_max_offset_ww'] / EA
		partials['moor_tension_max_offset_ww', 'maxval_fairlead'] = -1.

		partials['eff_length_max_offset_ww', 'z_moor'] = 1.
		partials['eff_length_max_offset_ww', 'water_depth'] = 1.
		partials['eff_length_max_offset_ww', 'EA_moor'] = mu * 9.80665 * l_eff_ww**2. / (2. * EA**2.)
		partials['eff_length_max_offset_ww', 'mass_dens_moor'] = -9.80665 * l_eff_ww**2. / (2. * EA) - ((np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.) + t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * (-2. * l_eff_ww**2. / t_star_ww**3.)) * (-outputs['moor_tension_max_offset_ww'] / (mu**2. * 9.80665))
		partials['eff_length_max_offset_ww', 'len_hor_moor'] = 0.
		partials['eff_length_max_offset_ww', 'len_tot_moor'] = 0.
		partials['eff_length_max_offset_ww', 'moor_tension_max_offset_ww'] = -((np.sqrt(1. + (l_eff_ww / t_star_ww)**2.) - 1.) + t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * (-2. * l_eff_ww**2. / t_star_ww**3.)) * 1. / (mu * 9.80665)
		partials['eff_length_max_offset_ww', 'eff_length_max_offset_ww'] = -mu * 9.80665 * l_eff_ww / EA - t_star_ww * (0.5 / np.sqrt(1. + (l_eff_ww / t_star_ww)**2.)) * 2. * l_eff_ww / t_star_ww**2.
		partials['eff_length_max_offset_ww', 'maxval_fairlead'] = 0.

		partials['maxval_fairlead', 'moor_tension_max_offset_ww'] = 1. / (mu * 9.80665) * np.arcsinh(mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww']) + outputs['moor_tension_max_offset_ww'] / (mu * 9.80665) * 1. / np.sqrt(1. + (mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww'])**2.) * (-mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww']**2.) + outputs['eff_length_max_offset_ww'] / EA
		partials['maxval_fairlead', 'eff_length_max_offset_ww'] = outputs['moor_tension_max_offset_ww'] / (mu * 9.80665) * 1. / np.sqrt(1. + (mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww'])**2.) * (mu * 9.80665 / outputs['moor_tension_max_offset_ww']) + outputs['moor_tension_max_offset_ww'] / EA
		partials['maxval_fairlead', 'mass_dens_moor'] = -outputs['moor_tension_max_offset_ww'] / (mu**2. * 9.80665) * np.arcsinh(mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww']) + outputs['moor_tension_max_offset_ww'] / (mu * 9.80665) * 1. / np.sqrt(1. + (mu * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww'])**2.) * 9.80665 * outputs['eff_length_max_offset_ww'] / outputs['moor_tension_max_offset_ww']
		partials['maxval_fairlead', 'EA_moor'] = -outputs['moor_tension_max_offset_ww'] * outputs['eff_length_max_offset_ww'] / EA**2.
		partials['maxval_fairlead', 'len_hor_moor'] = -1.