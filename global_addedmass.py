import numpy as np

from openmdao.api import ExplicitComponent

class GlobalAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('A11', val=0.)
		self.add_input('A15', val=0.)
		self.add_input('A55', val=0.)

		self.add_output('A_global', val=np.zeros((3,3)), units='kg')

	def compute(self, inputs, outputs):
		A11 = inputs['A11']
		A15 = inputs['A15']
		A55 = inputs['A55']

		outputs['A_global'] = np.zeros((3,3)) #Added mass for bending mode included in M17, M57, etc. TODO: separate

		outputs['A_global'][0,0] += A11
		outputs['A_global'][0,1] += A15
		outputs['A_global'][1,0] += A15
		outputs['A_global'][1,1] += A55