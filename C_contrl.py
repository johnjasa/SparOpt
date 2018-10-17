import numpy as np

from openmdao.api import ExplicitComponent

class Ccontrl(ExplicitComponent):

	def setup(self):
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)

		self.add_output('C_contrl', val=np.zeros((2,2)))

	def compute(self, inputs, outputs):
		k_i = inputs['k_i']
		k_p = inputs['k_p']
		eta = inputs['gain_corr_factor']

		outputs['C_contrl'] = np.array([[0., 0.],[eta * k_i, eta * k_p]])