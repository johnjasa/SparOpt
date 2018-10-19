import numpy as np

from openmdao.api import ExplicitComponent

class MooringStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('dmoor_tension_ww_dx', val=0., units='N/m')
		self.add_input('dmoor_tension_lw_dx', val=0., units='N/m')

		self.add_output('K_moor', val=0., units='N/m')

		self.declare_partials('K_moor', 'dmoor_tension_offset_ww_dx', val=1.)
		self.declare_partials('K_moor', 'dmoor_tension_offset_lw_dx', val=-0.5)

	def compute(self, inputs, outputs):
		outputs['K_moor'] = inputs['dmoor_tension_ww_dx'] - 0.5 * inputs['dmoor_tension_lw_dx']