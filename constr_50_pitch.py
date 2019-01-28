import numpy as np

from openmdao.api import ExplicitComponent

class Constr50Pitch(ExplicitComponent):

	def setup(self):
		self.add_input('maxval_pitch', val=0., units='rad')
		self.add_input('prob_max_pitch', val=0., units='rad')

		self.add_output('constr_50_pitch', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		maxval_pitch = inputs['maxval_pitch']
		prob_max_pitch = inputs['prob_max_pitch']

		outputs['constr_50_pitch'] = maxval_pitch / prob_max_pitch - 1.
	
	def compute_partials(self, inputs, partials):
		maxval_pitch = inputs['maxval_pitch']
		prob_max_pitch = inputs['prob_max_pitch']

		partials['constr_50_pitch', 'maxval_pitch'] = 1. / prob_max_pitch
		partials['constr_50_pitch', 'prob_max_pitch'] = -maxval_pitch / prob_max_pitch**2.