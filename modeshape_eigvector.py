import numpy as np
from scipy.linalg import det, eig, solve

from openmdao.api import ExplicitComponent

class ModeshapeEigvector(ExplicitComponent):

	def setup(self):
		self.add_input('A_eig', val=np.zeros((48,48)))

		self.add_output('eig_vector', val=np.ones(48), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		A = inputs['A_eig']

		eig_vals, eig_vecs = np.linalg.eig(A)

		#print 2. * np.pi / np.sqrt(eig_vals[-3:])
		
		outputs['eig_vector'] = eig_vecs[:,-3]

	def compute_partials(self, inputs, partials):
		A = inputs['A_eig']

		partials['eig_vector', 'A_eig'] = np.zeros((len(A),A.size))

		eig_vals, eig_vecs = np.linalg.eig(A)
		eig_val = eig_vals[-3]
		eig_vec = eig_vecs[:,-3]

		E = np.zeros_like(A)
		F = np.zeros_like(A)

		for i in xrange(len(A)):
			for j in xrange(len(A)):
				E[i,j] = eig_vals[j] - eig_vals[i]
				if i != j:
					F[i,j] = 1. / (eig_vals[j] - eig_vals[i])

		for i in xrange(len(A)):
			for j in xrange(len(A)):
				dA = np.zeros_like(A)
				dA[i,j] = 1.

				P = solve(eig_vecs, np.dot(dA, eig_vecs))

				dU = np.dot(eig_vecs, (F * P))

				partials['eig_vector', 'A_eig'][:,len(A)*i+j] = dU[:,-3]