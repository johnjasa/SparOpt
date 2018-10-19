import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class BstrExt(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('dthrust_dbldpitch', val=0., units='N/rad')
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('Bstr_ext', val=np.zeros((3,2)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		dthrust_dbldpitch = inputs['dthrust_dbldpitch'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		residuals['Bstr_ext'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Bstr_ext']) - np.array([[0., dthrust_dbldpitch],[0., CoG_rotor * dthrust_dbldpitch],[0., dthrust_dbldpitch]])

	def solve_nonlinear(self, inputs, outputs):
		dthrust_dbldpitch = inputs['dthrust_dbldpitch'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		outputs['Bstr_ext'] = np.matmul(np.linalg.inv(inputs['M_global'] + inputs['A_global']), np.array([[0., dthrust_dbldpitch],[0., CoG_rotor * dthrust_dbldpitch],[0., dthrust_dbldpitch]]))