import numpy as np

from openmdao.api import ExplicitComponent

class BallastElem(ExplicitComponent):

	def setup(self):
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_moor_zero', val=0., units='kg')
		self.add_input('rho_ball', val=0., units='kg/m**3')
		self.add_input('wt_ball', val=0., units='m')

		self.add_output('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_output('L_ball_elem', val=np.zeros(10), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		buoy_spar = inputs['buoy_spar']
		tot_M_spar = inputs['tot_M_spar']
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		M_turb = inputs['M_turb']
		M_moor_zero = inputs['M_moor_zero']
		rho_ball = inputs['rho_ball']
		wt_ball = inputs['wt_ball']

		M_ball = (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor_zero))

		L_ball_elem = np.zeros(len(D_spar))
		M_ball_elem = np.zeros(len(D_spar))

		accum_mass = 0.

		for i in xrange(len(D_spar)):
			if accum_mass < M_ball:
				if (accum_mass + np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * L_spar[i] * rho_ball) < M_ball:
					L_ball_elem[i] = L_spar[i]
					M_ball_elem[i] = np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * L_spar[i] * rho_ball
					accum_mass += M_ball_elem[i]
				else:
					L_ball_elem[i] += (M_ball - accum_mass) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					M_ball_elem[i] += M_ball - accum_mass
					accum_mass += M_ball_elem[i]

		outputs['L_ball_elem'] = L_ball_elem
		outputs['M_ball_elem'] = M_ball_elem

	def compute_partials(self, inputs, partials):
		buoy_spar = inputs['buoy_spar']
		tot_M_spar = inputs['tot_M_spar']
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		M_turb = inputs['M_turb']
		M_moor_zero = inputs['M_moor_zero']
		rho_ball = inputs['rho_ball']
		wt_ball = inputs['wt_ball']

		partials['L_ball_elem', 'buoy_spar'] = np.zeros((10,1))
		partials['L_ball_elem', 'tot_M_spar'] = np.zeros((10,1))
		partials['L_ball_elem', 'D_spar'] = np.zeros((10,10))
		partials['L_ball_elem', 'L_spar'] = np.zeros((10,10))
		partials['L_ball_elem', 'M_turb'] = np.zeros((10,1))
		partials['L_ball_elem', 'M_moor_zero'] = np.zeros((10,1))
		partials['L_ball_elem', 'rho_ball'] = np.zeros((10,1))
		partials['L_ball_elem', 'wt_ball'] = np.zeros((10,1))

		partials['M_ball_elem', 'buoy_spar'] = np.zeros((10,1))
		partials['M_ball_elem', 'tot_M_spar'] = np.zeros((10,1))
		partials['M_ball_elem', 'D_spar'] = np.zeros((10,10))
		partials['M_ball_elem', 'L_spar'] = np.zeros((10,10))
		partials['M_ball_elem', 'M_turb'] = np.zeros((10,1))
		partials['M_ball_elem', 'M_moor_zero'] = np.zeros((10,1))
		partials['M_ball_elem', 'rho_ball'] = np.zeros((10,1))
		partials['M_ball_elem', 'wt_ball'] = np.zeros((10,1))

		M_ball = (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor_zero))

		L_ball_elem = np.zeros(len(D_spar))
		M_ball_elem = np.zeros(len(D_spar))

		accum_mass = 0.

		for i in xrange(len(D_spar)):
			if accum_mass < M_ball:
				if (accum_mass + np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * L_spar[i] * rho_ball) < M_ball:
					L_ball_elem[i] = L_spar[i]
					M_ball_elem[i] = np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * L_spar[i] * rho_ball
					accum_mass += M_ball_elem[i]

					partials['L_ball_elem', 'L_spar'][i,i] += 1.

					partials['M_ball_elem', 'D_spar'][i,i] += np.pi / 2. * (D_spar[i] - 2. * wt_ball) * L_spar[i] * rho_ball
					partials['M_ball_elem', 'L_spar'][i,i] += np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball
					partials['M_ball_elem', 'rho_ball'][i,i] += np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * L_spar[i]
					partials['M_ball_elem', 'wt_ball'][i,i] += -np.pi * (D_spar[i] - 2. * wt_ball) * L_spar[i] * rho_ball

				else:
					L_ball_elem[i] += (M_ball - accum_mass) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					M_ball_elem[i] += M_ball - accum_mass
					accum_mass += M_ball_elem[i]

					for j in xrange(i):
						partials['L_ball_elem', 'D_spar'][i,j] += -(np.pi / 2. * (D_spar[j] - 2. * wt_ball) * L_spar[j] * rho_ball) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
						partials['L_ball_elem', 'L_spar'][i,j] += -(np.pi / 4. * (D_spar[j] - 2. * wt_ball)**2. * rho_ball) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
						partials['L_ball_elem', 'wt_ball'][i,0] += (np.pi * (D_spar[j] - 2. * wt_ball) * L_spar[j] * rho_ball) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
						partials['L_ball_elem', 'rho_ball'][i,0] += -(np.pi / 4. * (D_spar[j] - 2. * wt_ball)**2. * L_spar[j]) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)

						partials['M_ball_elem', 'D_spar'][i,j] += -np.pi / 4. * (D_spar[j] - 2. * wt_ball)**2. * L_spar[j] * rho_ball
						partials['M_ball_elem', 'L_spar'][i,j] += -np.pi / 4. * (D_spar[j] - 2. * wt_ball)**2. * rho_ball
						partials['M_ball_elem', 'wt_ball'][i,0] += np.pi * (D_spar[j] - 2. * wt_ball) * L_spar[j] * rho_ball
						partials['M_ball_elem', 'rho_ball'][i,0] += -np.pi / 4. * (D_spar[j] - 2. * wt_ball)**2. * L_spar[j]
					
					partials['L_ball_elem', 'buoy_spar'][i,0] += 9.80665 / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					partials['L_ball_elem', 'tot_M_spar'][i,0] += -1. / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					partials['L_ball_elem', 'M_turb'][i,0] += -1. / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					partials['L_ball_elem', 'M_moor_zero'][i,0] += -1. / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					partials['L_ball_elem', 'D_spar'][i,i] += -2. * (M_ball - accum_mass) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**3. * rho_ball)
					partials['L_ball_elem', 'rho_ball'][i,0] += -(M_ball - accum_mass) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball**2.) - np.sum(np.pi / 4. * (D_spar[:i] - 2. * wt_ball)**2. * L_spar[:i]) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)
					partials['L_ball_elem', 'wt_ball'][i,0] += 4. * (M_ball - accum_mass) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**3. * rho_ball) + np.sum(np.pi * (D_spar[:i] - 2. * wt_ball) * L_spar[:i] * rho_ball) / (np.pi / 4. * (D_spar[i] - 2. * wt_ball)**2. * rho_ball)

					partials['M_ball_elem', 'buoy_spar'][i,0] += 9.80665
					partials['M_ball_elem', 'tot_M_spar'][i,0] += -1.
					partials['M_ball_elem', 'M_turb'][i,0] += -1.
					partials['M_ball_elem', 'M_moor_zero'][i,0] += -1.
					partials['M_ball_elem', 'rho_ball'][i,0] += -np.sum(np.pi / 4. * (D_spar[:i] - 2. * wt_ball)**2. * L_spar[:i])
					partials['M_ball_elem', 'wt_ball'][i,0] += np.sum(np.pi * (D_spar[:i] - 2. * wt_ball) * L_spar[:i] * rho_ball)