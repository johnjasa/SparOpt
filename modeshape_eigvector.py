import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import det, inv

from openmdao.api import ImplicitComponent

class ModeshapeEigvector(ImplicitComponent):

	def setup(self):
		self.add_input('K_mode', val=np.zeros((34,34)), units='N/m')
		self.add_input('M_mode', val=np.zeros((34,34)), units='kg')

		self.add_output('omega_eig', val=2., units='rad/s')
		self.add_output('eig_vector', val=np.ones(34), units='m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		K = inputs['K_mode']
		M = inputs['M_mode']

		residuals['omega_eig'] = det(K - outputs['omega_eig']**2. * M)
		residuals['eig_vector'] = (K - outputs['omega_eig']**2. * M).dot(outputs['eig_vector'])

	def solve_nonlinear(self, inputs, outputs):
		K = inputs['K_mode']
		M = inputs['M_mode']

		eig_vals, eig_vecs = eigs(K, k=3, M=M, sigma=(2.*np.pi/500.)**2.)

		outputs['omega_eig'] = np.real(np.sqrt(eig_vals[-1]))
		
		outputs['eig_vector'] = np.real(eig_vecs[:,-1])

	def linearize(self, inputs, outputs, partials):
		K = inputs['K_mode']
		M = inputs['M_mode']

		eig_matrix = K - outputs['omega_eig']**2. * M

		#residuals['omega_eig'] = det(K - outputs['omega_eig']**2. * M)
		#residuals['eig_vector'] = (K - outputs['omega_eig']**2. * M).dot(outputs['eig_vector'])

		partials['omega_eig', 'K_mode'] = det(eig_matrix) * (2. * inv(eig_matrix) - np.multiply(inv(eig_matrix),np.identity(34))).flatten()
		partials['omega_eig', 'M_mode'] = -outputs['omega_eig']**2. * det(eig_matrix) * (2. * inv(eig_matrix) - np.multiply(inv(eig_matrix),np.identity(34))).flatten()
		partials['omega_eig', 'omega_eig'] = np.sum(det(eig_matrix) * (2. * inv(eig_matrix) - np.multiply(inv(eig_matrix),np.identity(34))).flatten() * -2. * outputs['omega_eig'] * M.flatten())
		#partials['omega_eig', 'eig_vector'] = 

		#partials['eig_vector', 'K_mode'] = 
		#partials['eig_vector', 'M_mode'] = 
		#partials['eig_vector', 'omega_eig'] = 
		#partials['eig_vector', 'eig_vector'] = 