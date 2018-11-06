import numpy as np

from openmdao.api import Group

from aero_loads import AeroLoads
from wind_speed import WindSpeed
from total_aero_loads import TotalAeroLoads

class Aero(Group):

	def initialize(self):
		self.options.declare('blades', types=dict)
		self.options.declare('freqs', types=dict)

	def setup(self):
		blades = self.options['blades']
		freqs = self.options['freqs']

		self.add_subsystem('aero_loads', AeroLoads(blades=blades), promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0'], promotes_outputs=['b_elem_r', 'b_elem_dr', 'Fn_0', 'Ft_0', 'dFn_dv', 'dFn_dbldpitch', 'dFn_drotspeed', 'dFt_dv', 'dFt_dbldpitch', 'dFt_drotspeed'])

		self.add_subsystem('wind_speed', WindSpeed(blades=blades, freqs=freqs), promotes_inputs=['windspeed_0', 'rotspeed_0', 'b_elem_r', 'b_elem_dr', 'dFn_dv', 'dFt_dv'], promotes_outputs=['thrust_wind', 'moment_wind', 'torque_wind'])

		self.add_subsystem('total_aero_loads', TotalAeroLoads(blades=blades), promotes_inputs=['b_elem_r', 'b_elem_dr', 'Fn_0', 'Ft_0', 'dFn_dv', 'dFt_dv', 'dFn_drotspeed', 'dFt_drotspeed', 'dFn_dbldpitch', 'dFt_dbldpitch'], promotes_outputs=['thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch'])