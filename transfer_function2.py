import numpy as np
from scipy import linalg

from openmdao.api import ExplicitComponent

class TransferFunction(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_IA_inv', val=np.zeros((N_omega,11,11)))
		self.add_input('Im_IA_inv', val=np.zeros((N_omega,11,11)))
		self.add_input('B_feedbk', val=np.zeros((11,6)))

		self.add_output('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_output('Im_H_feedbk', val=np.zeros((N_omega,11,6)))

		Acols = Acols1 = np.tile(np.arange(11),6)
		for i in xrange(1,N_omega*11):
			Acols = np.concatenate((Acols,Acols1 + np.ones(len(Acols1)) * 11 * i),0)

		Bcols = Bcols1 = np.arange(0,66,6)
		for i in xrange(1,6):
			Bcols = np.concatenate((Bcols,Bcols1 + np.ones(len(Bcols1)) * i),0)

		Bcols = np.tile(Bcols, N_omega*11)

		self.declare_partials('Re_H_feedbk', 'Re_IA_inv', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Acols)
		self.declare_partials('Re_H_feedbk', 'Im_IA_inv', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Acols)
		self.declare_partials('Re_H_feedbk', 'B_feedbk', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Bcols)
		self.declare_partials('Im_H_feedbk', 'Re_IA_inv', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Acols)
		self.declare_partials('Im_H_feedbk', 'Im_IA_inv', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Acols)
		self.declare_partials('Im_H_feedbk', 'B_feedbk', rows=np.repeat(np.arange(11*N_omega*6),11), cols=Bcols)

	def compute(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			IA_inv = inputs['Re_IA_inv'][i] + 1j * inputs['Im_IA_inv'][i]
			H_feedbk = np.matmul(IA_inv, inputs['B_feedbk'])

			outputs['Re_H_feedbk'][i] = np.real(H_feedbk)
			outputs['Im_H_feedbk'][i] = np.imag(H_feedbk)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		dH_dA = inputs['B_feedbk'].T.flatten()

		for i in xrange(N_omega*11):
			partials['Re_H_feedbk', 'Re_IA_inv'][i*11*6:i*11*6+11*6] = dH_dA
			partials['Im_H_feedbk', 'Im_IA_inv'][i*11*6:i*11*6+11*6] = dH_dA

		for i in xrange(N_omega):
			for j in xrange(11):
				partials['Re_H_feedbk', 'B_feedbk'][i*11*11*6+j*11*6:i*11*11*6+j*11*6+11*6] = np.tile(inputs['Re_IA_inv'][i,j],6)
				partials['Im_H_feedbk', 'B_feedbk'][i*11*11*6+j*11*6:i*11*11*6+j*11*6+11*6] = np.tile(inputs['Im_IA_inv'][i,j],6)