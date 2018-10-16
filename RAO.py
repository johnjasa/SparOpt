import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class RAO(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)))
		self.add_input('A_global', val=np.zeros((3,3)))
		self.add_input('B_global', val=np.zeros((3,3)))
		self.add_input('K_global', val=np.zeros((3,3)))
		self.add_input('omega_wave', val=np.zeros(100), units='rad/s')
		self.add_input('Re_wave_forces', val=np.zeros((100,3,1)))
		self.add_input('Im_wave_forces', val=np.zeros((100,3,1)))

		self.add_output('Re_RAO', val=np.zeros((100,3,1)))
		self.add_output('Im_RAO', val=np.zeros((100,3,1)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		#R = Ax - b.

		for i in xrange(100):
			residuals['Re_RAO'][i] = np.real((-inputs['omega_wave'][i]**2. * (inputs['M_global'] + inputs['A_global']) + 1j * inputs['omega_wave'][i] * inputs['B_global'] + inputs['K_global']).dot(outputs['Re_RAO'][i] + 1j * outputs['Im_RAO'][i])) - inputs['Re_wave_forces']
			residuals['Im_RAO'][i] = np.imag((-inputs['omega_wave'][i]**2. * (inputs['M_global'] + inputs['A_global']) + 1j * inputs['omega_wave'][i] * inputs['B_global'] + inputs['K_global']).dot(outputs['Re_RAO'][i] + 1j * outputs['Im_RAO'][i])) - inputs['Im_wave_forces']

	def solve_nonlinear(self, inputs, outputs):

		for i in xrange(100):
			RAO = linalg.solve(-inputs['omega_wave'][i]**2. * (inputs['M_global'] + inputs['A_global']) + 1j * inputs['omega_wave'][i] * inputs['B_global'] + inputs['K_global'], inputs['Re_wave_forces'][i] + 1j * inputs['Im_wave_forces'][i])

			outputs['Re_RAO'][i] = np.real(RAO)
			outputs['Im_RAO'][i] = np.imag(RAO)