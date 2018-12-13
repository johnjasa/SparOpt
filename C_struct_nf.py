import numpy as np

from openmdao.api import ExplicitComponent

class Cstruct(ExplicitComponent):

	def setup(self):
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('C_struct', val=np.zeros((2,7)))

		self.declare_partials('C_struct', 'CoG_rotor', val=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]))

	def compute(self, inputs, outputs):
		outputs['C_struct'] = np.array([[0., 0., 0., 0., 0., 0., 1.],[0., 0., 0., 1., inputs['CoG_rotor'], 1., 0.]])