import numpy as np

from openmdao.api import ExplicitComponent

class MooringGenMass(ExplicitComponent):

	def setup(self):
		self.add_input('mass_dens_moor', val=0., units='kg/m')
		self.add_input('norm_r_moor', val=np.zeros(100), units='m')
		self.add_input('eff_length_offset_ww', val=0., units='m')

		self.add_output('gen_m_moor', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mu = inputs['mass_dens_moor']
		norm_r = inputs['norm_r_moor']
		L = inputs['eff_length_offset_ww']

		dL = L / 100.

		outputs['gen_m_moor'] = np.sum(mu * dL * norm_r**2.)

	def compute_partials(self, inputs, partials): #TODO
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']