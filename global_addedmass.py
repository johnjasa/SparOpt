import numpy as np

from openmdao.api import ExplicitComponent

class GlobalAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('A11', val=0., units='kg')
		self.add_input('A15', val=0., units='kg*m')
		self.add_input('A17', val=0., units='kg')
		self.add_input('A55', val=0., units='kg*m**2')
		self.add_input('A57', val=0., units='kg*m')
		self.add_input('A77', val=0., units='kg')

		self.add_output('A_global', val=np.zeros((3,3)), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		A11 = inputs['A11']
		A15 = inputs['A15']
		A17 = inputs['A17']
		A55 = inputs['A55']
		A57 = inputs['A57']
		A77 = inputs['A77']

		outputs['A_global'] = np.zeros((3,3))

		outputs['A_global'][0,0] += A11
		outputs['A_global'][0,1] += A15
		outputs['A_global'][0,2] += A17
		outputs['A_global'][1,0] += A15
		outputs['A_global'][1,1] += A55
		outputs['A_global'][1,2] += A57
		outputs['A_global'][2,0] += A17
		outputs['A_global'][2,1] += A57
		outputs['A_global'][2,2] += A77

	def compute_partials(self, inputs, partials):
		partials['A_global', 'A11'] = np.concatenate((np.array([1., 0., 0.]),np.zeros((2,3))),0)
		partials['A_global', 'A15'] = np.concatenate((np.array([0., 1., 0.]),np.array([1., 0., 0.]),np.zeros((1,3))),0)
		partials['A_global', 'A17'] = np.concatenate((np.array([0., 0., 1.]),np.zeros((1,3)),np.array([1., 0., 0.])),0)
		partials['A_global', 'A55'] = np.concatenate((np.zeros((1,3)),np.array([0., 1., 0.]),np.zeros((1,3))),0)
		partials['A_global', 'A57'] = np.concatenate((np.zeros((1,3)),np.array([0., 0., 1.]),np.array([0., 1., 0.])),0)
		partials['A_global', 'A77'] = np.concatenate((np.zeros((2,3)),np.array([0., 0., 1.])),0)