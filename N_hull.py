import numpy as np

from openmdao.api import ExplicitComponent

class NHull(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('N_hull', val=np.zeros(10), units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		M_ball = inputs['M_ball']
		L_ball = inputs['L_ball']
		spar_draft = inputs['spar_draft']
		z_ball = -spar_draft + L_ball[0]
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']

		outputs['N_hull'][0] = 1025. * 9.80665 * Z_spar[0] * np.pi / 4. * D_spar_p[0]**2.

		flag = 0

		for i in xrange(1,len(D_spar_p)-1):
			outputs['N_hull'][i] = outputs['N_hull'][i-1] + M_spar[i-1] * 9.80665 - 1025. * 9.80665 * Z_spar[i] * np.pi / 4. * (D_spar_p[i-1]**2. - D_spar_p[i]**2.)

			if Z_spar[i] >= z_ball and Z_spar[i-1] < z_ball:
				outputs['N_hull'][i] += M_ball / L_ball * (z_ball - Z_spar[i-1]) * 9.80665
			elif Z_spar[i] < z_ball:
				outputs['N_hull'][i] += M_ball / L_ball * (Z_spar[i] - Z_spar[i-1]) * 9.80665

			if Z_spar[i] >= z_moor and flag == 0:
				outputs['N_hull'][i] += M_moor * 9.80665
				flag = 1

	def compute_partials(self, inputs, partials):
		D_spar_p = inputs['D_spar_p']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		M_ball = inputs['M_ball']
		L_ball = inputs['L_ball']
		spar_draft = inputs['spar_draft']
		z_ball = -spar_draft + L_ball[0]
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']

		partials['N_hull', 'D_spar_p'] = np.zeros((10,11))
		partials['N_hull', 'Z_spar'] = np.zeros((10,11))
		partials['N_hull', 'M_spar'] = np.zeros((10,10))
		partials['N_hull', 'M_ball'] = np.zeros((10,1))
		partials['N_hull', 'L_ball'] = np.zeros((10,1))
		partials['N_hull', 'spar_draft'] = np.zeros((10,1))
		partials['N_hull', 'M_moor'] = np.zeros((10,1))
		partials['N_hull', 'z_moor'] = np.zeros((10,1))

		partials['N_hull', 'D_spar_p'][0,0] += 1025. * 9.80665 * Z_spar[0] * np.pi / 2. * D_spar_p[0]
		partials['N_hull', 'Z_spar'][0,0] += 1025. * 9.80665 * np.pi / 4. * D_spar_p[0]**2.

		flag = 0

		for i in xrange(1,len(D_spar_p)-1):
			partials['N_hull', 'D_spar_p'][i,:] += partials['N_hull', 'D_spar_p'][i-1,:]
			partials['N_hull', 'Z_spar'][i,:] += partials['N_hull', 'Z_spar'][i-1,:]
			partials['N_hull', 'M_spar'][i,:] += partials['N_hull', 'M_spar'][i-1,:]
			partials['N_hull', 'M_ball'][i] += partials['N_hull', 'M_ball'][i-1]
			partials['N_hull', 'L_ball'][i] += partials['N_hull', 'L_ball'][i-1]
			partials['N_hull', 'spar_draft'][i] += partials['N_hull', 'spar_draft'][i-1]
			partials['N_hull', 'M_moor'][i] += partials['N_hull', 'M_moor'][i-1]

			partials['N_hull', 'D_spar_p'][i,i-1] += -1025. * 9.80665 * Z_spar[i] * np.pi / 2. * D_spar_p[i-1]
			partials['N_hull', 'D_spar_p'][i,i] += 1025. * 9.80665 * Z_spar[i] * np.pi / 2. * D_spar_p[i]
			partials['N_hull', 'Z_spar'][i,i] += -1025. * 9.80665 * np.pi / 4. * (D_spar_p[i-1]**2. - D_spar_p[i]**2.)
			partials['N_hull', 'M_spar'][i,i-1] += 9.80665

			if Z_spar[i] >= z_ball and Z_spar[i-1] < z_ball:
				partials['N_hull', 'Z_spar'][i,i-1] += -M_ball / L_ball * 9.80665
				partials['N_hull', 'M_ball'][i,0] += 1. / L_ball * (z_ball - Z_spar[i-1]) * 9.80665
				partials['N_hull', 'L_ball'][i,0] += -M_ball / L_ball**2. * (z_ball - Z_spar[i-1]) * 9.80665 + M_ball / L_ball * 9.80665
				partials['N_hull', 'spar_draft'][i,0] += -M_ball / L_ball * 9.80665
			elif Z_spar[i] < z_ball:
				partials['N_hull', 'Z_spar'][i,i-1] += -M_ball / L_ball * 9.80665
				partials['N_hull', 'Z_spar'][i,i] += M_ball / L_ball * 9.80665
				partials['N_hull', 'M_ball'][i,0] += 1. / L_ball * (Z_spar[i] - Z_spar[i-1]) * 9.80665
				partials['N_hull', 'L_ball'][i,0] += -M_ball / L_ball**2. * (Z_spar[i] - Z_spar[i-1]) * 9.80665

			if Z_spar[i] >= z_moor and flag == 0:
				partials['N_hull', 'M_moor'][i,0] += 9.80665
				flag = 1