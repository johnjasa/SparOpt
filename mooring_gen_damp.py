import numpy as np

from openmdao.api import ExplicitComponent

class MooringGenDamp(ExplicitComponent):

	def setup(self):
		self.add_input('norm_r_moor', val=np.zeros(100), units='m')
		self.add_input('Cd_moor', val=0.)
		self.add_input('D_moor', val=0., units='m')
		self.add_input('eff_length_offset_ww', val=0., units='m')

		self.add_output('gen_c_moor', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		norm_r = inputs['norm_r_moor']
		Cd = inputs['Cd_moor']
		D = inputs['D_moor']
		L = inputs['eff_length_offset_ww']

		dL = L / 100.

		outputs['gen_c_moor'] = np.sum(0.5 * 1025. * Cd * D * dL * norm_r**2.)

	def compute_partials(self, inputs, partials): #TODO
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']