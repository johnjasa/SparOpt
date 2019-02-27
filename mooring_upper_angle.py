import numpy as np

from openmdao.api import ExplicitComponent

class MooringUpperAngle(ExplicitComponent):

	def setup(self):
		self.add_input('mass_dens_moor', val=0., units='m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_ww', val=0., units='m')

		self.add_output('phi_upper_end', val=0., units='rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']

		outputs['phi_upper_end'] = np.arctan(mu * 9.80665 * L / H) #angle with the horizontal plane

	def compute_partials(self, inputs, partials): #TODO check
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']
	
		partials['phi_upper_end', 'mass_dens_moor'] = 1. / (1. + (mu * 9.80665 * L / H)**2.) * (9.80665 * L / H)
		partials['phi_upper_end', 'moor_tension_offset_ww'] = 1. / (1. + (mu * 9.80665 * L / H)**2.) * (-mu * 9.80665 * L / H**2.)
		partials['phi_upper_end', 'eff_length_offset_ww'] = 1. / (1. + (mu * 9.80665 * L / H)**2.) * (mu * 9.80665 / H)