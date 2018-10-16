import numpy as np

from openmdao.api import ExplicitComponent

class TotalAeroLoads(ExplicitComponent):

	def initialize(self):
		self.options.declare('blades', types=dict)

	def setup(self):
		blades = self.options['blades']
		self.N_b_elem = blades['N_b_elem']

		self.add_input('b_elem_r', val=np.zeros(self.N_b_elem), units='m')
		self.add_input('b_elem_dr', val=0., units='m')
		self.add_input('Fn_0', val=np.zeros(self.N_b_elem), units='N/m')
		self.add_input('Ft_0', val=np.zeros(self.N_b_elem), units='N/m')
		self.add_input('dFn_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')
		self.add_input('dFt_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')
		self.add_input('dFn_drotspeed', val=np.zeros(self.N_b_elem), units='N*s/(m*rad)')
		self.add_input('dFt_drotspeed', val=np.zeros(self.N_b_elem), units='N*s/(m*rad)')
		self.add_input('dFn_dbldpitch', val=np.zeros(self.N_b_elem), units='N/(m*rad)')
		self.add_input('dFt_dbldpitch', val=np.zeros(self.N_b_elem), units='N/(m*rad)')

		self.add_output('thrust_0', val=0., units='N')
		self.add_output('torque_0', val=0., units='N*m')
		self.add_output('dthrust_dv', val=0., units='N*s/m')
		self.add_output('dmoment_dv', val=0., units='N*s')
		self.add_output('dtorque_dv', val=0., units='N*s')
		self.add_output('dthrust_drotspeed', val=0., units='N*s/rad')
		self.add_output('dtorque_drotspeed', val=0., units='N*m*s/rad')
		self.add_output('dthrust_dbldpitch', val=0., units='N/rad')
		self.add_output('dtorque_dbldpitch', val=0., units='N*m/rad')

	def compute(self,inputs,outputs):
		outputs['thrust_0'] = 3. * np.sum(inputs['Fn_0'] * inputs['b_elem_dr'])
		outputs['torque_0'] = 3. * np.sum(inputs['Ft_0'] * inputs['b_elem_r'] * inputs['b_elem_dr'])
		outputs['dthrust_dv'] = 3. * np.sum(inputs['dFn_dv'] * inputs['b_elem_dr'])
		outputs['dmoment_dv'] = 3. / 2. * np.sum(inputs['dFn_dv'] * inputs['b_elem_r'] * inputs['b_elem_dr'])
		outputs['dtorque_dv'] = 3. * np.sum(inputs['dFt_dv'] * inputs['b_elem_r'] * inputs['b_elem_dr'])
		outputs['dthrust_drotspeed'] = 3. * np.sum(inputs['dFn_drotspeed'] * inputs['b_elem_dr'])
		outputs['dtorque_drotspeed'] = 3. * np.sum(inputs['dFt_drotspeed'] * inputs['b_elem_r'] * inputs['b_elem_dr'])
		outputs['dthrust_dbldpitch'] = 3. * np.sum(inputs['dFn_dbldpitch'] * inputs['b_elem_dr'])
		outputs['dtorque_dbldpitch'] = 3. * np.sum(inputs['dFt_dbldpitch'] * inputs['b_elem_r'] * inputs['b_elem_dr'])