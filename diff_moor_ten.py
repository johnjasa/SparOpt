import numpy as np
from scipy.optimize import root, fsolve, least_squares

from openmdao.api import ImplicitComponent

class DiffMoorTen(ImplicitComponent):

	def setup(self):
		self.add_input('EA_moor', val=1., units='N')
		self.add_input('mass_dens_moor', val=1., units='kg/m')
		self.add_input('eff_length_offset_ww', val=1., units='m')
		self.add_input('moor_tension_offset_ww', val=1., units='N')
		self.add_input('eff_length_offset_lw', val=1., units='m')
		self.add_input('moor_tension_offset_lw', val=1., units='N')

		self.add_output('deff_length_ww_dx', val=1., units='m/m')
		self.add_output('dmoor_tension_ww_dx', val=1., units='N/m')
		self.add_output('deff_length_lw_dx', val=1., units='m/m')
		self.add_output('dmoor_tension_lw_dx', val=1., units='N/m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']

		l_eff_ww = inputs['eff_length_offset_ww']
		t_hor_ww = inputs['moor_tension_offset_ww']
		l_eff_lw = inputs['eff_length_offset_lw']
		t_hor_lw = inputs['moor_tension_offset_lw']

		dl_eff_ww_dx = outputs['deff_length_ww_dx']
		dt_hor_ww_dx = outputs['dmoor_tension_ww_dx']
		dl_eff_lw_dx = outputs['deff_length_lw_dx']
		dt_hor_lw_dx = outputs['dmoor_tension_lw_dx']
		
		residuals['dmoor_tension_ww_dx'] = -dl_eff_ww_dx - 1. + 1. / (mu * 9.80665) * dt_hor_ww_dx * np.arcsinh(l_eff_ww * mu * 9.80665 / t_hor_ww) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + dt_hor_ww_dx * l_eff_ww / EA + t_hor_ww / EA * dl_eff_ww_dx
		residuals['deff_length_ww_dx'] = -mu * 9.80665 * l_eff_ww / EA * dl_eff_ww_dx - 1. / (mu * 9.80665) * dt_hor_ww_dx * (np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) - 1.) - t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx)

		residuals['dmoor_tension_lw_dx'] = -dl_eff_lw_dx + 1. + 1. / (mu * 9.80665) * dt_hor_lw_dx * np.arcsinh(l_eff_lw * mu * 9.80665 / t_hor_lw) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + dt_hor_lw_dx * l_eff_lw / EA + t_hor_lw / EA * dl_eff_lw_dx
		residuals['deff_length_lw_dx'] = -mu * 9.80665 * l_eff_lw / EA * dl_eff_lw_dx - 1. / (mu * 9.80665) * dt_hor_lw_dx * (np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) - 1.) - t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx)

	def solve_nonlinear(self, inputs, outputs):
		EA = inputs['EA_moor'][0]
		mu = inputs['mass_dens_moor'][0]

		l_eff_ww = inputs['eff_length_offset_ww'][0]
		t_hor_ww = inputs['moor_tension_offset_ww'][0]
		l_eff_lw = inputs['eff_length_offset_lw'][0]
		t_hor_lw = inputs['moor_tension_offset_lw'][0]

		def fun_ww(x):
			return [-x[0] - 1. + 1. / (mu * 9.80665) * x[1] * np.arcsinh(l_eff_ww * mu * 9.80665 / t_hor_ww) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * x[0] - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * x[1]) + x[1] * l_eff_ww / EA + t_hor_ww / EA * x[0],\
			-mu * 9.80665 * l_eff_ww / EA * x[0] - 1. / (mu * 9.80665) * x[1] * (np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) - 1.) - t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww) * (mu * 9.80665 / t_hor_ww * x[0] - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * x[1])]

		def fun_lw(x):
			return [-x[0] + 1. + 1. / (mu * 9.80665) * x[1] * np.arcsinh(l_eff_lw * mu * 9.80665 / t_hor_lw) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * x[0] - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * x[1]) + x[1] * l_eff_lw / EA + t_hor_lw / EA * x[0],\
			-mu * 9.80665 * l_eff_lw / EA * x[0] - 1. / (mu * 9.80665) * x[1] * (np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) - 1.) - t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw) * (mu * 9.80665 / t_hor_lw * x[0] - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * x[1])]

		#sol_ww = fsolve(fun_ww, [1.0, 1.0e3], xtol=1e-5)
		sol_ww = least_squares(fun_ww, [1.0, 1.0e3], bounds=(np.array([0.,0.]),np.array([np.inf,np.inf])), xtol=1e-6)

		#sol_lw = fsolve(fun_lw, [-1.0, -1.0e3], xtol=1e-5)
		sol_lw = least_squares(fun_lw, [-1.0, -1.0e3], bounds=(np.array([-np.inf,-np.inf]),np.array([0.,0.])), xtol=1e-6)

		outputs['deff_length_ww_dx'] = sol_ww.x[0]
		outputs['dmoor_tension_ww_dx'] = sol_ww.x[1]
		outputs['deff_length_lw_dx'] = sol_lw.x[0]
		outputs['dmoor_tension_lw_dx'] = sol_lw.x[1]

	def linearize(self, inputs, outputs, partials):
		EA = inputs['EA_moor']
		mu = inputs['mass_dens_moor']

		l_eff_ww = inputs['eff_length_offset_ww']
		t_hor_ww = inputs['moor_tension_offset_ww']
		l_eff_lw = inputs['eff_length_offset_lw']
		t_hor_lw = inputs['moor_tension_offset_lw']

		dl_eff_ww_dx = outputs['deff_length_ww_dx']
		dt_hor_ww_dx = outputs['dmoor_tension_ww_dx']
		dl_eff_lw_dx = outputs['deff_length_lw_dx']
		dt_hor_lw_dx = outputs['dmoor_tension_lw_dx']

		partials['dmoor_tension_ww_dx', 'EA_moor'] = -dt_hor_ww_dx * l_eff_ww / EA**2. - t_hor_ww / EA**2. * dl_eff_ww_dx
		partials['dmoor_tension_ww_dx', 'mass_dens_moor'] = -1. / (mu**2. * 9.80665) * dt_hor_ww_dx * np.arcsinh(l_eff_ww * mu * 9.80665 / t_hor_ww) + 1. / (mu * 9.80665) * dt_hor_ww_dx * 1. / np.sqrt(1. + (l_eff_ww * mu * 9.80665 / t_hor_ww)**2.) * l_eff_ww * 9.80665 / t_hor_ww - t_hor_ww / (mu**2. * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) - t_hor_ww / (mu * 9.80665) * 1. / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * mu * ((l_eff_ww * 9.80665) / t_hor_ww)**2. * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx)
		partials['dmoor_tension_ww_dx', 'eff_length_offset_ww'] = 1. / (mu * 9.80665) * dt_hor_ww_dx * 1. / np.sqrt(1. + (l_eff_ww * mu * 9.80665 / t_hor_ww)**2.) * mu * 9.80665 / t_hor_ww - t_hor_ww / (mu * 9.80665) * 1. / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * l_eff_ww * ((mu * 9.80665) / t_hor_ww)**2. * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (-mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + dt_hor_ww_dx / EA
		partials['dmoor_tension_ww_dx', 'moor_tension_offset_ww'] = -1. / (mu * 9.80665) * dt_hor_ww_dx * 1. / np.sqrt(1. + (l_eff_ww * mu * 9.80665 / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww**2.) + 1. / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + t_hor_ww / (mu * 9.80665) * 1. / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * ((l_eff_ww * mu * 9.80665) / t_hor_ww) * ((l_eff_ww * mu * 9.80665) / t_hor_ww**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (-mu * 9.80665 / t_hor_ww**2. * dl_eff_ww_dx + 2. * l_eff_ww * mu * 9.80665 / t_hor_ww**3. * dt_hor_ww_dx) + 1. / EA * dl_eff_ww_dx
		partials['dmoor_tension_ww_dx', 'eff_length_offset_lw'] = 0.
		partials['dmoor_tension_ww_dx', 'moor_tension_offset_lw'] = 0.
		partials['dmoor_tension_ww_dx', 'deff_length_ww_dx'] = -1. + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww) + t_hor_ww / EA
		partials['dmoor_tension_ww_dx', 'dmoor_tension_ww_dx'] = 1. / (mu * 9.80665) * np.arcsinh(l_eff_ww * mu * 9.80665 / t_hor_ww) + t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (-l_eff_ww * mu * 9.80665 / t_hor_ww**2.) + l_eff_ww / EA
		partials['dmoor_tension_ww_dx', 'deff_length_lw_dx'] = 0.
		partials['dmoor_tension_ww_dx', 'dmoor_tension_lw_dx'] = 0.

		partials['deff_length_ww_dx', 'EA_moor'] = mu * 9.80665 * l_eff_ww / EA**2. * dl_eff_ww_dx 
		partials['deff_length_ww_dx', 'mass_dens_moor'] = -9.80665 * l_eff_ww / EA * dl_eff_ww_dx + 1. / (mu**2. * 9.80665) * dt_hor_ww_dx * (np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) - 1.) - 1. / (mu * 9.80665) * dt_hor_ww_dx * (1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * ((l_eff_ww * 9.80665) / t_hor_ww)**2. * mu) - l_eff_ww / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + l_eff_ww / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * ((l_eff_ww * 9.80665) / t_hor_ww)**2. * mu * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx)
		partials['deff_length_ww_dx', 'eff_length_offset_ww'] = -mu * 9.80665 / EA * dl_eff_ww_dx - 1. / (mu * 9.80665) * dt_hor_ww_dx * (1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)) * ((mu * 9.80665) / t_hor_ww)**2. * l_eff_ww - 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) + 1. / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * (mu * 9.80665 / t_hor_ww)**2. * l_eff_ww**2. * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) - 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * l_eff_ww * (-mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx)
		partials['deff_length_ww_dx', 'moor_tension_offset_ww'] = 1. / (mu * 9.80665) * dt_hor_ww_dx * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * ((l_eff_ww * mu * 9.80665) / t_hor_ww) * ((l_eff_ww * mu * 9.80665) / t_hor_ww**2.) - l_eff_ww / (1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.)**(3. / 2.) * ((l_eff_ww * mu * 9.80665) / t_hor_ww) * ((l_eff_ww * mu * 9.80665) / t_hor_ww**2.) * (mu * 9.80665 / t_hor_ww * dl_eff_ww_dx - l_eff_ww * mu * 9.80665 / t_hor_ww**2. * dt_hor_ww_dx) - l_eff_ww / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (-mu * 9.80665 / t_hor_ww**2. * dl_eff_ww_dx + 2. * l_eff_ww * mu * 9.80665 / t_hor_ww**3. * dt_hor_ww_dx)
		partials['deff_length_ww_dx', 'eff_length_offset_lw'] = 0.
		partials['deff_length_ww_dx', 'moor_tension_offset_lw'] = 0.
		partials['deff_length_ww_dx', 'deff_length_ww_dx'] = -mu * 9.80665 * l_eff_ww / EA - t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww) * (mu * 9.80665 / t_hor_ww)
		partials['deff_length_ww_dx', 'dmoor_tension_ww_dx'] = -1. / (mu * 9.80665) * (np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) - 1.) - t_hor_ww / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_ww * mu * 9.80665) / t_hor_ww)**2.) * (l_eff_ww * mu * 9.80665 / t_hor_ww) * (-l_eff_ww * mu * 9.80665 / t_hor_ww**2.)
		partials['deff_length_ww_dx', 'deff_length_lw_dx'] = 0.
		partials['deff_length_ww_dx', 'dmoor_tension_lw_dx'] = 0.

		partials['dmoor_tension_lw_dx', 'EA_moor'] = -dt_hor_lw_dx * l_eff_lw / EA**2. - t_hor_lw / EA**2. * dl_eff_lw_dx
		partials['dmoor_tension_lw_dx', 'mass_dens_moor'] = -1. / (mu**2. * 9.80665) * dt_hor_lw_dx * np.arcsinh(l_eff_lw * mu * 9.80665 / t_hor_lw) + 1. / (mu * 9.80665) * dt_hor_lw_dx * 1. / np.sqrt(1. + (l_eff_lw * mu * 9.80665 / t_hor_lw)**2.) * l_eff_lw * 9.80665 / t_hor_lw - t_hor_lw / (mu**2. * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) - t_hor_lw / (mu * 9.80665) * 1. / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * mu * ((l_eff_lw * 9.80665) / t_hor_lw)**2. * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx)
		partials['dmoor_tension_lw_dx', 'eff_length_offset_ww'] = 0.
		partials['dmoor_tension_lw_dx', 'moor_tension_offset_ww'] = 0.
		partials['dmoor_tension_lw_dx', 'eff_length_offset_lw'] = 1. / (mu * 9.80665) * dt_hor_lw_dx * 1. / np.sqrt(1. + (l_eff_lw * mu * 9.80665 / t_hor_lw)**2.) * mu * 9.80665 / t_hor_lw - t_hor_lw / (mu * 9.80665) * 1. / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * l_eff_lw * ((mu * 9.80665) / t_hor_lw)**2. * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (-mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + dt_hor_lw_dx / EA
		partials['dmoor_tension_lw_dx', 'moor_tension_offset_lw'] = -1. / (mu * 9.80665) * dt_hor_lw_dx * 1. / np.sqrt(1. + (l_eff_lw * mu * 9.80665 / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw**2.) + 1. / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + t_hor_lw / (mu * 9.80665) * 1. / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * ((l_eff_lw * mu * 9.80665) / t_hor_lw) * ((l_eff_lw * mu * 9.80665) / t_hor_lw**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (-mu * 9.80665 / t_hor_lw**2. * dl_eff_lw_dx + 2. * l_eff_lw * mu * 9.80665 / t_hor_lw**3. * dt_hor_lw_dx) + 1. / EA * dl_eff_lw_dx
		partials['dmoor_tension_lw_dx', 'deff_length_ww_dx'] = 0.
		partials['dmoor_tension_lw_dx', 'dmoor_tension_ww_dx'] = 0.
		partials['dmoor_tension_lw_dx', 'deff_length_lw_dx'] = -1. + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw) + t_hor_lw / EA
		partials['dmoor_tension_lw_dx', 'dmoor_tension_lw_dx'] = 1. / (mu * 9.80665) * np.arcsinh(l_eff_lw * mu * 9.80665 / t_hor_lw) + t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (-l_eff_lw * mu * 9.80665 / t_hor_lw**2.) + l_eff_lw / EA

		partials['deff_length_lw_dx', 'EA_moor'] = -mu * 9.80665 * l_eff_lw / EA**2. * dl_eff_lw_dx 
		partials['deff_length_lw_dx', 'mass_dens_moor'] = -9.80665 * l_eff_lw / EA * dl_eff_lw_dx + 1. / (mu**2. * 9.80665) * dt_hor_lw_dx * (np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) - 1.) - 1. / (mu * 9.80665) * dt_hor_lw_dx * (1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * ((l_eff_lw * 9.80665) / t_hor_lw)**2. * mu) - l_eff_lw / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + l_eff_lw / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * ((l_eff_lw * 9.80665) / t_hor_lw)**2. * mu * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx)
		partials['deff_length_lw_dx', 'eff_length_offset_ww'] = 0.
		partials['deff_length_lw_dx', 'moor_tension_offset_ww'] = 0.
		partials['deff_length_lw_dx', 'eff_length_offset_lw'] = -mu * 9.80665 / EA * dl_eff_lw_dx - 1. / (mu * 9.80665) * dt_hor_lw_dx * (1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)) * ((mu * 9.80665) / t_hor_lw)**2. * l_eff_lw - 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) + 1. / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * (mu * 9.80665 / t_hor_lw)**2. * l_eff_lw**2. * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) - 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * l_eff_lw * (-mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx)
		partials['deff_length_lw_dx', 'moor_tension_offset_lw'] = 1. / (mu * 9.80665) * dt_hor_lw_dx * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * ((l_eff_lw * mu * 9.80665) / t_hor_lw) * ((l_eff_lw * mu * 9.80665) / t_hor_lw**2.) - l_eff_lw / (1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.)**(3. / 2.) * ((l_eff_lw * mu * 9.80665) / t_hor_lw) * ((l_eff_lw * mu * 9.80665) / t_hor_lw**2.) * (mu * 9.80665 / t_hor_lw * dl_eff_lw_dx - l_eff_lw * mu * 9.80665 / t_hor_lw**2. * dt_hor_lw_dx) - l_eff_lw / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (-mu * 9.80665 / t_hor_lw**2. * dl_eff_lw_dx + 2. * l_eff_lw * mu * 9.80665 / t_hor_lw**3. * dt_hor_lw_dx)
		partials['deff_length_lw_dx', 'deff_length_ww_dx'] = 0.
		partials['deff_length_lw_dx', 'dmoor_tension_ww_dx'] = 0.
		partials['deff_length_lw_dx', 'deff_length_lw_dx'] = -mu * 9.80665 * l_eff_lw / EA - t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw) * (mu * 9.80665 / t_hor_lw)
		partials['deff_length_lw_dx', 'dmoor_tension_lw_dx'] = -1. / (mu * 9.80665) * (np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) - 1.) - t_hor_lw / (mu * 9.80665) * 1. / np.sqrt(1. + ((l_eff_lw * mu * 9.80665) / t_hor_lw)**2.) * (l_eff_lw * mu * 9.80665 / t_hor_lw) * (-l_eff_lw * mu * 9.80665 / t_hor_lw**2.)