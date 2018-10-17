import numpy as np

from openmdao.api import ExplicitComponent

class MooringMass(ExplicitComponent):

	def setup(self):
		self.add_output('eff_length_offset_ww', val=0., units='m')
		self.add_output('eff_length_offset_lw', val=0., units='m')

		self.add_output('M_moor', val=0., units='kg')

	def compute(self, inputs, outputs):
		mu = 155.41

		outputs['M_moor'] = outputs['eff_length_offset_ww'] + 2. * outputs['eff_length_offset_lw']