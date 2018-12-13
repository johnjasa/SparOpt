import numpy as np

from openmdao.api import ExplicitComponent

class MeanSurge(ExplicitComponent):

	def setup(self):
		self.add_input('moor_offset', val=0., units='m')
		self.add_input('mean_pitch', val=0., units='rad')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('mean_surge', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		moor_offset = inputs['moor_offset']
		mean_pitch = inputs['mean_pitch']
		z_moor = inputs['z_moor']

		outputs['mean_surge'] = moor_offset - np.sin(mean_pitch) * z_moor

	def compute_partials(self, inputs, partials):
		moor_offset = inputs['moor_offset']
		mean_pitch = inputs['mean_pitch']
		z_moor = inputs['z_moor']

		partials['mean_surge', 'moor_offset'] = 1.
		partials['mean_surge', 'mean_pitch'] = -np.cos(mean_pitch) * z_moor
		partials['mean_surge', 'z_moor'] = -np.sin(mean_pitch)