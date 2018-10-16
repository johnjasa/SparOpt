import numpy as np

from openmdao.api import Group

from A_struct import Astruct
from B_struct import Bstruct
from C_struct import Cstruct
from A_contrl import Acontrl
from B_contrl import Bcontrl
from C_contrl import Ccontrl
from A_feedbk import Afeedbk
from B_feedbk import Bfeedbk
from transfer_function import TransferFunction

class StateSpace(Group):

	def setup(self):
		self.add_subsystem('A_struct', Astruct(), promotes_inputs=['M_global', 'A_global', 'B_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed'], promotes_outputs=['A_struct'])

		self.add_subsystem('B_struct', Bstruct(), promotes_inputs=['M_global', 'A_global', 'CoG_rotor', 'I_d', 'dthrust_dbldpitch', 'dtorque_dbldpitch'], promotes_outputs=['B_struct'])

		self.add_subsystem('C_struct', Cstruct(), promotes_outputs=['C_struct'])

		self.add_subsystem('A_contrl', Acontrl(), promotes_inputs=['omega_lowpass'], promotes_outputs=['A_contrl'])

		self.add_subsystem('B_contrl', Bcontrl(), promotes_inputs=['omega_lowpass'], promotes_outputs=['B_contrl'])

		self.add_subsystem('C_contrl', Ccontrl(), promotes_inputs=['k_i', 'k_p'], promotes_outputs=['C_contrl'])

		self.add_subsystem('A_feedbk', Afeedbk(), promotes_inputs=['A_struct', 'A_contrl', 'B_struct', 'B_contrl', 'C_struct', 'C_contrl'], promotes_outputs=['A_feedbk'])

		self.add_subsystem('B_feedbk', Bfeedbk(), promotes_inputs=['M_global', 'A_global', 'CoG_rotor', 'psi_d_top', 'I_d', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv'], promotes_outputs=['B_feedbk'])

		self.add_subsystem('transfer_function', TransferFunction(), promotes_inputs=['A_feedbk', 'B_feedbk', 'omega'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])