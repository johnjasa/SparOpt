import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeHydrostiff(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')

		self.add_output('K_hydrostatic', val=0., units='N*m/rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		tot_M_spar = inputs['tot_M_spar']
		CoG_spar = inputs['CoG_spar']
		M_ball = inputs['M_ball']
		CoG_ball = inputs['CoG_ball']
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']

		outputs['K_hydrostatic'] = buoy_spar * CoB - 9.80665 * (tot_M_spar * CoG_spar + M_ball * CoG_ball + M_moor * z_moor - 1025. * np.pi/64. * D_spar[-1]**4.)

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		tot_M_spar = inputs['tot_M_spar']
		CoG_spar = inputs['CoG_spar']
		M_ball = inputs['M_ball']
		CoG_ball = inputs['CoG_ball']
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']

		partials['K_hydrostatic', 'D_spar'] = np.zeros((1,10))
		partials['K_hydrostatic', 'D_spar'][0,-1] = 9.80665 * 1025. * np.pi/16. * D_spar[-1]**3.
		partials['K_hydrostatic', 'tot_M_spar'] = -9.80665 * CoG_spar
		partials['K_hydrostatic', 'CoG_spar'] = -9.80665 * tot_M_spar
		partials['K_hydrostatic', 'M_ball'] = -9.80665 * CoG_ball
		partials['K_hydrostatic', 'CoG_ball'] = -9.80665 * M_ball
		partials['K_hydrostatic', 'M_moor'] = -9.80665 * z_moor
		partials['K_hydrostatic', 'z_moor'] = -9.80665 * M_moor
		partials['K_hydrostatic', 'buoy_spar'] = CoB
		partials['K_hydrostatic', 'CoB'] = buoy_spar