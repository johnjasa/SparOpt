import numpy as np

from openmdao.api import ExplicitComponent

class Ballast(ExplicitComponent):

	def setup(self):
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('rho_ball', val=0., units='kg/m**3')
		self.add_input('wt_ball', val=0., units='m')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('M_ball', val=0., units='kg')
		self.add_output('CoG_ball', val=0., units='m')
		self.add_output('L_ball', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		buoy_spar = inputs['buoy_spar']
		tot_M_spar = inputs['tot_M_spar']
		spar_draft = inputs['spar_draft']
		D_spar  = inputs['D_spar']
		M_turb = inputs['M_turb']
		M_moor = inputs['M_moor']
		rho_ball = inputs['rho_ball']
		wt_ball = inputs['wt_ball']

		outputs['L_ball'] = (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)

		outputs['M_ball'] = (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor))
		outputs['CoG_ball'] =  -spar_draft + 0.5 * (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)

	def compute_partials(self, inputs, partials):
		buoy_spar = inputs['buoy_spar'][0]
		tot_M_spar = inputs['tot_M_spar'][0]
		spar_draft = inputs['spar_draft'][0]
		D_spar  = inputs['D_spar']
		M_turb = inputs['M_turb'][0]
		M_moor = inputs['M_moor'][0]
		rho_ball = inputs['rho_ball'][0]
		wt_ball = inputs['wt_ball'][0]

		partials['L_ball', 'buoy_spar'] = (1. / 9.80665) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)
		partials['L_ball', 'tot_M_spar'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)
		partials['L_ball', 'D_spar'] = np.array([-2. * (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**3. * rho_ball), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		partials['L_ball', 'M_turb'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)
		partials['L_ball', 'M_moor'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball)
		partials['L_ball', 'rho_ball'] = -(buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball**2.)
		partials['L_ball', 'wt_ball'] = 4. * (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**3. * rho_ball)
		partials['L_ball', 'spar_draft'] = 0.

		partials['M_ball', 'buoy_spar'] = 1. / 9.80665
		partials['M_ball', 'tot_M_spar'] = -1.
		partials['M_ball', 'D_spar'] = np.zeros(10)
		partials['M_ball', 'M_turb'] = -1.
		partials['M_ball', 'M_moor'] = -1.
		partials['M_ball', 'rho_ball'] = 0.
		partials['M_ball', 'wt_ball'] = 0.
		partials['M_ball', 'spar_draft'] = 0.

		partials['CoG_ball', 'buoy_spar'] = (1. / 9.80665) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball) * 0.5
		partials['CoG_ball', 'tot_M_spar'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball) * 0.5
		partials['CoG_ball', 'D_spar'] = np.array([-(buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**3. * rho_ball), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		partials['CoG_ball', 'M_turb'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball) * 0.5
		partials['CoG_ball', 'M_moor'] = -1. / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball) * 0.5
		partials['CoG_ball', 'rho_ball'] = -(buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**2. * rho_ball**2.) * 0.5
		partials['CoG_ball', 'wt_ball'] = 2. * (buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_spar[0] - 2. * wt_ball)**3. * rho_ball)
		partials['CoG_ball', 'spar_draft'] = -1.