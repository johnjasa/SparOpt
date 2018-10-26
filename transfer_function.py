import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class TransferFunction(ImplicitComponent):

	def setup(self):
		self.add_input('A_feedbk', val=np.zeros((9,9)))
		self.add_input('B_feedbk', val=np.zeros((9,6)))
		self.add_input('omega', val=np.zeros(3493), units='rad/s')

		self.add_output('Re_H_feedbk', val=np.ones((3493,9,6)))
		self.add_output('Im_H_feedbk', val=np.ones((3493,9,6)))

		#self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		omega = inputs['omega']
		N_omega = len(omega)

		for i in xrange(N_omega):
			residuals['Re_H_feedbk'][i] = np.real((1j*omega[i] * np.identity(9) - inputs['A_feedbk']).dot(outputs['Re_H_feedbk'][i] + 1j * outputs['Im_H_feedbk'][i])) - inputs['B_feedbk']
			residuals['Im_H_feedbk'][i] = np.imag((1j*omega[i] * np.identity(9) - inputs['A_feedbk']).dot(outputs['Re_H_feedbk'][i] + 1j * outputs['Im_H_feedbk'][i])) - inputs['B_feedbk']

	def solve_nonlinear(self, inputs, outputs):
		omega = inputs['omega']
		N_omega = len(omega)

		for i in xrange(N_omega):
			H_feedbk = np.matmul(linalg.inv(1j*omega[i] * np.identity(9) - inputs['A_feedbk']), inputs['B_feedbk'])

			outputs['Re_H_feedbk'][i] = np.real(H_feedbk)
			outputs['Im_H_feedbk'][i] = np.imag(H_feedbk)

	def linearize(self, inputs, outputs, partials):
		omega = inputs['omega']
		N_omega = len(omega)

		#for i in xrange(N_omega):
			#partials['Re_H_feedbk', 'A_feedbk'][i] = np.kron(np.identity(9),outputs['Re_H_feedbk'][i])
			#partials['Re_H_feedbk', 'B_feedbk'][i] = 
			#partials['Re_H_feedbk', 'Re_H_feedbk'][i] = 
			#partials['Re_H_feedbk', 'Im_H_feedbk'][i] = 

			#partials['Im_H_feedbk', 'A_feedbk'][i] = 
			#partials['Im_H_feedbk', 'B_feedbk'][i] = 
			#partials['Im_H_feedbk', 'Re_H_feedbk'][i] = 
			#partials['Im_H_feedbk', 'Im_H_feedbk'][i] = 