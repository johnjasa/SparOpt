import numpy as np

from openmdao.api import ExplicitComponent

class GlobalMass(ExplicitComponent):

	def setup(self):
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('CoG_turb', val=0., units='m')
		self.add_input('I_spar', val=0., units='kg*m**2')
		self.add_input('I_ball', val=0., units='kg*m**2')
		self.add_input('I_turb', val=0., units='kg*m**2')
		self.add_input('M17', val=0., units='kg')
		self.add_input('M57', val=0., units='kg*m')
		self.add_input('M77', val=0., units='kg')

		self.add_output('M_global', val=np.zeros((3,3)), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		tot_M_spar = inputs['tot_M_spar']
		M_ball = inputs['M_ball']
		M_turb = inputs['M_turb']
		CoG_spar = inputs['CoG_spar']
		CoG_ball = inputs['CoG_ball']
		CoG_turb = inputs['CoG_turb']
		I_spar = inputs['I_spar']
		I_ball = inputs['I_ball']
		I_turb = inputs['I_turb']
		M17 = inputs['M17']
		M57 = inputs['M57']
		M77 = inputs['M77']

		outputs['M_global'] = np.zeros((3,3))

		outputs['M_global'][0,0] += M_turb + tot_M_spar + M_ball
		outputs['M_global'][0,1] += M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball
		outputs['M_global'][0,2] += M17
		outputs['M_global'][1,0] += M_turb * CoG_turb + tot_M_spar * CoG_spar  + M_ball * CoG_ball
		outputs['M_global'][1,1] += I_turb + I_spar + I_ball
		outputs['M_global'][1,2] += M57
		outputs['M_global'][2,0] += M17
		outputs['M_global'][2,1] += M57
		outputs['M_global'][2,2] += M77

	def compute_partials(self, inputs, partials):
		tot_M_spar = inputs['tot_M_spar']
		M_ball = inputs['M_ball']
		M_turb = inputs['M_turb']
		CoG_spar = inputs['CoG_spar']
		CoG_ball = inputs['CoG_ball']
		CoG_turb = inputs['CoG_turb']

		partials['M_global', 'tot_M_spar'] = np.array([1., CoG_spar, 0., CoG_spar, 0., 0., 0., 0., 0.])
		partials['M_global', 'M_ball'] = np.array([1., CoG_ball, 0., CoG_ball, 0., 0., 0., 0., 0.])
		partials['M_global', 'M_turb'] = np.array([1., CoG_turb, 0., CoG_turb, 0., 0., 0., 0., 0.])
		partials['M_global', 'CoG_spar'] = np.array([0., tot_M_spar, 0., tot_M_spar, 0., 0., 0., 0., 0.])
		partials['M_global', 'CoG_ball'] = np.array([0., M_ball, 0., M_ball, 0., 0., 0., 0., 0.])
		partials['M_global', 'CoG_turb'] = np.array([0., M_turb, 0., M_turb, 0., 0., 0., 0., 0.])
		partials['M_global', 'I_spar'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
		partials['M_global', 'I_ball'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
		partials['M_global', 'I_turb'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
		partials['M_global', 'M17'] = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0.])
		partials['M_global', 'M57'] = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0.])
		partials['M_global', 'M77'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])