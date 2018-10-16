import numpy as np

from openmdao.api import ExplicitComponent

class Ballast(ExplicitComponent):

	def setup(self):
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('M_secs', val=np.zeros(3), units='kg')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('rho_ball', val=0., units='kg/m**3')
		self.add_input('wt_ball', val=0., units='m')

		self.add_output('M_ball', val=0., units='kg')
		self.add_output('CoG_ball', val=0., units='m')
		self.add_output('L_ball', val=0., units='m')

	def compute(self, inputs, outputs):
		buoy_spar = inputs['buoy_spar']
		M_spar = np.sum(inputs['M_secs'])
		draft_spar = np.sum(inputs['L_secs']) - 10. #tower base 10m above SWL
		D_secs  = inputs['D_secs']
		M_turb = np.sum(inputs['M_tower']) + inputs['M_nacelle'] + inputs['M_rotor']
		M_moor = 330000.#inputs['M_moor']
		rho_ball = inputs['rho_ball']
		wt_ball = inputs['wt_ball']

		outputs['L_ball'] = (buoy_spar / 9.80665 - (M_spar + M_turb + M_moor)) / (np.pi / 4. * (D_secs[-1] - 2. * wt_ball)**2. * rho_ball) #height of ballast, m above inner wall spar bottom

		outputs['M_ball'] = np.pi / 4. * (D_secs[-1] - 2. * wt_ball)**2. * outputs['L_ball'] * rho_ball
		outputs['CoG_ball'] = wt_ball + outputs['L_ball'] / 2. - draft_spar