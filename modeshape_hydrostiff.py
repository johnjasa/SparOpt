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
