import numpy as np
from scipy import linalg

from openmdao.api import ExplicitComponent

class TransferFunctionPreInv(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_IA', val=np.zeros((N_omega,9,9)))
		self.add_input('Im_IA', val=np.zeros((N_omega,9,9)))

		self.add_output('Re_IA_inv', val=np.zeros((N_omega,9,9)))
		self.add_output('Im_IA_inv', val=np.zeros((N_omega,9,9)))

		Acols = Acols1 = np.tile(np.arange(9*9), 9*9)
		for i in xrange(1,N_omega):
			Acols = np.concatenate((Acols,Acols1 + np.ones(len(Acols1)) * 9 * 9 * i),0)

		self.declare_partials('Re_IA_inv', 'Re_IA', rows=np.repeat(np.arange(9*9*N_omega),9*9), cols=Acols)
		self.declare_partials('Re_IA_inv', 'Im_IA', rows=np.repeat(np.arange(9*9*N_omega),9*9), cols=Acols)
		self.declare_partials('Im_IA_inv', 'Re_IA', rows=np.repeat(np.arange(9*9*N_omega),9*9), cols=Acols)
		self.declare_partials('Im_IA_inv', 'Im_IA', rows=np.repeat(np.arange(9*9*N_omega),9*9), cols=Acols)

	def compute(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			IA = inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]
			IA_inv = linalg.inv(IA)

			outputs['Re_IA_inv'][i] = np.real(IA_inv)
			outputs['Im_IA_inv'][i] = np.imag(IA_inv)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			partials['Re_IA_inv', 'Re_IA'][i*(9*9)**2:i*(9*9)**2+(9*9)**2] = np.real(np.kron(np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]),-np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]).T)).flatten()
			partials['Re_IA_inv', 'Im_IA'][i*(9*9)**2:i*(9*9)**2+(9*9)**2] = -np.imag(np.kron(np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]),-np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]).T)).flatten()

			partials['Im_IA_inv', 'Re_IA'][i*(9*9)**2:i*(9*9)**2+(9*9)**2] = np.imag(np.kron(np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]),-np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]).T)).flatten()
			partials['Im_IA_inv', 'Im_IA'][i*(9*9)**2:i*(9*9)**2+(9*9)**2] = np.real(np.kron(np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]),-np.linalg.inv(inputs['Re_IA'][i] + 1j * inputs['Im_IA'][i]).T)).flatten()