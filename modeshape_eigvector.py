import numpy as np
from scipy.linalg import det, eig, solve

from openmdao.api import ExplicitComponent

class ModeshapeEigvector(ExplicitComponent):

	def setup(self):
		self.add_input('A_eig', val=np.zeros((46,46)))
		self.add_input('struct_damp_ratio', val=0.)

		self.add_output('eig_vector', val=np.ones(46), units='m')
		self.add_output('alpha_damp', val=0., units='s')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		A = inputs['A_eig']
		struct_damp_ratio = inputs['struct_damp_ratio']

		eig_vals, eig_vecs = np.linalg.eig(A)

		#print eig_vals[-3:]
		#print 2. * np.pi / np.sqrt(eig_vals[-3:])
		
		outputs['eig_vector'] = eig_vecs[:,-3]
		if eig_vals[-3] <= 0.:
			outputs['alpha_damp'] = 0.007
		else:
			outputs['alpha_damp'] = 2. * struct_damp_ratio / np.sqrt(np.real(eig_vals[-3]))

	def compute_partials(self, inputs, partials):
		A = inputs['A_eig']
		struct_damp_ratio = inputs['struct_damp_ratio']

		partials['eig_vector', 'A_eig'] = np.zeros((len(A),A.size))
		partials['alpha_damp', 'A_eig'] = np.zeros((1,A.size))

		eig_vals, eig_vecs = np.linalg.eig(A)
		eig_val = eig_vals[-3]
		eig_vec = eig_vecs[:,-3]

		if eig_val <= 0.:
			partials['alpha_damp', 'struct_damp_ratio'] = 0.
		else:
			partials['alpha_damp', 'struct_damp_ratio'] = 2. / np.sqrt(np.real(eig_val))

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

				dD = np.diag(np.identity(len(A)) * P)
				dU = np.dot(eig_vecs, (F * P))

				partials['eig_vector', 'A_eig'][:,len(A)*i+j] = dU[:,-3]
				if eig_val <= 0.:
					partials['alpha_damp', 'A_eig'][0,len(A)*i+j] = 0.
				else:
					partials['alpha_damp', 'A_eig'][0,len(A)*i+j] = -struct_damp_ratio / np.real(eig_val)**(3./2.) * dD[-3]
