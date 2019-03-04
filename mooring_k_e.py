import numpy as np

from openmdao.api import ExplicitComponent

class MooringKe(ExplicitComponent):

	def setup(self):
		self.add_input('EA_moor', val=0., units='N')
		self.add_input('eff_length_offset_ww', val=0., units='m')

		self.add_output('k_e_moor', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		EA = inputs['EA_moor']
		L = inputs['eff_length_offset_ww']

		outputs['k_e_moor'] = EA / L

	def compute_partials(self, inputs, partials): #TODO check
		EA = inputs['EA_moor']
		L = inputs['eff_length_offset_ww']

		partials['k_e_moor', 'EA_moor'] = 1. / L
		partials['k_e_moor', 'eff_length_offset_ww'] = -EA / L**2.