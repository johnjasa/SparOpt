import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class TransferFunction(ImplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('A_feedbk', val=np.zeros((11,11)))
		self.add_input('B_feedbk', val=np.zeros((11,6)))

		self.add_output('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_output('Im_H_feedbk', val=np.zeros((N_omega,11,6)))

		Acols = Acols1 = np.tile(np.arange(11), 6)
		for i in xrange(1,11):
			Acols = np.concatenate((Acols,Acols1 + np.ones(11*6) * 11 * i),0)
		Acols = np.tile(Acols,N_omega)

		Hcols2 = Hcols1 = np.arange(0,11*6,6)
		for i in xrange(1,6):
			Hcols2 = np.concatenate((Hcols2,Hcols1 + np.ones(11) * i),0)
		Hcols = Hcols2 = np.tile(Hcols2,11)
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,Hcols2 + np.ones(11*11*6) * 11 * 6 * i),0)

		self.declare_partials('Re_H_feedbk', 'A_feedbk', rows=np.repeat(np.arange(N_omega*11*6), 11), cols=Acols)
		self.declare_partials('Re_H_feedbk', 'B_feedbk', rows=np.arange(N_omega*11*6), cols=np.tile(np.arange(66),N_omega))
		self.declare_partials('Re_H_feedbk', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega*11*6), 11), cols=Hcols)
		self.declare_partials('Re_H_feedbk', 'Im_H_feedbk', rows=np.arange(N_omega*11*6), cols=np.arange(N_omega*11*6))
		self.declare_partials('Im_H_feedbk', 'A_feedbk', rows=np.repeat(np.arange(N_omega*11*6), 11), cols=Acols)
		self.declare_partials('Im_H_feedbk', 'B_feedbk', rows=np.arange(N_omega*11*6), cols=np.tile(np.arange(66),N_omega))
		self.declare_partials('Im_H_feedbk', 'Re_H_feedbk', rows=np.arange(N_omega*11*6), cols=np.arange(N_omega*11*6))
		self.declare_partials('Im_H_feedbk', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega*11*6), 11), cols=Hcols)

	def apply_nonlinear(self, inputs, outputs, residuals):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			residuals['Re_H_feedbk'][i] = np.real((1j*omega[i] * np.identity(11) - inputs['A_feedbk']).dot(outputs['Re_H_feedbk'][i] + 1j * outputs['Im_H_feedbk'][i])) - inputs['B_feedbk']
			residuals['Im_H_feedbk'][i] = np.imag((1j*omega[i] * np.identity(11) - inputs['A_feedbk']).dot(outputs['Re_H_feedbk'][i] + 1j * outputs['Im_H_feedbk'][i])) - inputs['B_feedbk']

	def solve_nonlinear(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			H_feedbk = linalg.solve(1j*omega[i] * np.identity(11) - inputs['A_feedbk'], inputs['B_feedbk'])

			outputs['Re_H_feedbk'][i] = np.real(H_feedbk)
			outputs['Im_H_feedbk'][i] = np.imag(H_feedbk)

	def linearize(self, inputs, outputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)

		Hre_A = -np.tile(np.transpose(outputs['Re_H_feedbk'][0]).flatten(),11)
		Him_A = -np.tile(np.transpose(outputs['Im_H_feedbk'][0]).flatten(),11)

		Hre_Hre = np.tile(-inputs['A_feedbk'][0].flatten(),6)
		Hre_Him = -omega[0] * np.ones(66)

		for i in xrange(1,11):
			Hre_Hre = np.concatenate((Hre_Hre,np.tile(-inputs['A_feedbk'][i].flatten(),6)),0)

		Hre_Hre = np.tile(Hre_Hre,N_omega)

		for i in xrange(1,N_omega):
			Hre_A = np.concatenate((Hre_A,-np.tile(np.transpose(outputs['Re_H_feedbk'][i]).flatten(),11)),0)
			Him_A = np.concatenate((Him_A,-np.tile(np.transpose(outputs['Im_H_feedbk'][i]).flatten(),11)),0)

			Hre_Him = np.concatenate((Hre_Him,-omega[i] * np.ones(66)),0)

		partials['Re_H_feedbk', 'A_feedbk'] = Hre_A
		partials['Im_H_feedbk', 'A_feedbk'] = Him_A

		partials['Re_H_feedbk', 'B_feedbk'] = -np.ones(N_omega*11*6)
		partials['Im_H_feedbk', 'B_feedbk'] = -np.ones(N_omega*11*6)

		partials['Re_H_feedbk', 'Re_H_feedbk'] = Hre_Hre
		partials['Im_H_feedbk', 'Re_H_feedbk'] = -Hre_Him

		partials['Re_H_feedbk', 'Im_H_feedbk'] = Hre_Him
		partials['Im_H_feedbk', 'Im_H_feedbk'] = Hre_Hre