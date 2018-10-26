import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import det

from openmdao.api import ImplicitComponent

class Modeshape(ImplicitComponent):

	def setup(self):
		self.add_input('K_', val=np.zeros(10), units='m')
		self.add_input('M_', val=np.zeros(10), units='m')

		self.add_output('omega_eig', val=0., units='rad/s')
		self.add_output('eig_vector', val=np.zeros(34), units='m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		K = inputs['K_']
		M= inputs['M_']

		residuals['omega_eig'] = det(K - outputs['omega_eig']**2. * M)
		residuals['eig_vector'] = (K - outputs['omega_eig']**2. * M).dot(outputs['eig_vector'])

	def solve_nonlinear(self, inputs, outputs):
		K = inputs['K_']
		M= inputs['M_']

		eig_vals, eig_vecs = eigs(K, k=3, M=M, sigma=(2.*np.pi/500.)**2.)

		outputs['omega_eig'] = np.real(np.sqrt(eig_vals[-1]))
		
		outputs['eig_vector'] = np.real(eig_vecs[:,-1])