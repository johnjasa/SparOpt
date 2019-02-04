import numpy as np
from scipy.linalg import eig, solve

from openmdao.api import ExplicitComponent

class Poles(ExplicitComponent):

	def setup(self):
		self.add_input('A_feedbk', val=np.zeros((11,11)))

		self.add_output('poles', val=np.zeros(11))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		A = inputs['A_feedbk']

		outputs['poles'] = np.real(np.linalg.eig(A)[0])

	def compute_partials(self, inputs, partials):
		A = inputs['A_feedbk']

		eig_vals, eig_vecs = np.linalg.eig(A)

		for i in xrange(len(A)):
			for j in xrange(len(A)):
				dA = np.zeros_like(A)
				dA[i,j] = 1.

				P = solve(eig_vecs, np.dot(dA, eig_vecs))

				dD = np.diag(np.identity(len(A)) * P)

				partials['poles', 'A_feedbk'][:,len(A)*i+j] = np.real(dD)