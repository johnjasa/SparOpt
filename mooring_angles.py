import numpy as np

from openmdao.api import ExplicitComponent

class MooringAngles(ExplicitComponent):

	def setup(self):
		self.add_input('mass_dens_moor', val=0., units='m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_ww', val=0., units='m')

		self.add_output('phi_moor', val=np.zeros(100), units='rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']

		s = np.linspace(0.,L,100)

		outputs['phi_moor'] = np.arctan(mu * 9.80665 * s / H) #angle with the horizontal plane

	def compute_partials(self, inputs, partials): #TODO check
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']

		s = np.linspace(0.,L,100)

		partials['phi_moor', 'mass_dens_moor'][:,0] = 1. / (1. + (mu * 9.80665 * s / H)**2.) * (9.80665 * s / H)
		partials['phi_moor', 'moor_tension_offset_ww'][:,0] = 1. / (1. + (mu * 9.80665 * s / H)**2.) * (-mu * 9.80665 * s / H**2.)
		partials['phi_moor', 'eff_length_offset_ww'][:,0] = 1. / (1. + (mu * 9.80665 / H)**2.) * np.linspace(0.,1.,100)