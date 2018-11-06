import numpy as np

from openmdao.api import ExplicitComponent

class TotalCoG(ExplicitComponent):

	def setup(self):
		self.add_input('M_turb', val=1., units='kg')
		self.add_input('tot_M_spar', val=1., units='kg')
		self.add_input('M_ball', val=1., units='kg')
		self.add_input('CoG_turb', val=0., units='m')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('CoG_ball', val=0., units='m')

		self.add_output('CoG_total', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		M_turb = inputs['M_turb']
		tot_M_spar = inputs['tot_M_spar']
		M_ball = inputs['M_ball']
		CoG_turb = inputs['CoG_turb']
		CoG_spar = inputs['CoG_spar']
		CoG_ball = inputs['CoG_ball']

		outputs['CoG_total'] = (M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball) / (tot_M_spar + M_turb + M_ball)

	def compute_partials(self, inputs, partials):
		M_turb = inputs['M_turb']
		tot_M_spar = inputs['tot_M_spar']
		M_ball = inputs['M_ball']
		CoG_turb = inputs['CoG_turb']
		CoG_spar = inputs['CoG_spar']
		CoG_ball = inputs['CoG_ball']

		partials['CoG_total', 'M_turb'] = CoG_turb / (tot_M_spar + M_turb + M_ball) - (M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball) / (tot_M_spar + M_turb + M_ball)**2.
		partials['CoG_total', 'tot_M_spar'] = CoG_spar / (tot_M_spar + M_turb + M_ball) - (M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball) / (tot_M_spar + M_turb + M_ball)**2.
		partials['CoG_total', 'M_ball'] = CoG_ball / (tot_M_spar + M_turb + M_ball) - (M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball) / (tot_M_spar + M_turb + M_ball)**2.
		partials['CoG_total', 'CoG_turb'] = M_turb / (tot_M_spar + M_turb + M_ball)
		partials['CoG_total', 'CoG_spar'] = tot_M_spar / (tot_M_spar + M_turb + M_ball)
		partials['CoG_total', 'CoG_ball'] = M_ball / (tot_M_spar + M_turb + M_ball)