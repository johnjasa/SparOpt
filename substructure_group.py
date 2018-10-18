import numpy as np

from openmdao.api import Group

from spar_mass import SparMass
from spar_total_mass import SparTotalMass
from spar_inertia import SparInertia
from spar_addedmass import SparAddedMass
from spar_cog import SparCoG
from tower_mass import TowerMass
from tower_total_mass import TowerTotalMass
from tower_inertia import TowerInertia
from tower_cog import TowerCoG
from ballast import Ballast
from ball_inertia import BallInertia
from buoyancy import Buoyancy
from aero_damping import AeroDamping
from modeshape import Modeshape
from bending_mass import BendingMass
from bending_addedmass import BendingAddedMass
from bending_damping import BendingDamping
from bending_stiffness import BendingStiffness
from global_mass import GlobalMass
from global_addedmass import GlobalAddedMass
from global_damping import GlobalDamping
from global_stiffness import GlobalStiffness
from wave_loads import WaveLoads
#from RAO import RAO

class Substructure(Group):

	 def setup(self):
	 	self.add_subsystem('spar_mass', SparMass(), promotes_inputs=['D_spar', 'L_spar', 'wt_spar'], promotes_outputs=['M_spar'])

	 	self.add_subsystem('spar_total_mass', SparTotalMass(), promotes_inputs=['M_spar'], promotes_outputs=['tot_M_spar'])

	 	self.add_subsystem('spar_addedmass', SparAddedMass(), promotes_inputs=['D_spar', 'L_spar'], promotes_outputs=['A11', 'A15', 'A55'])

	 	self.add_subsystem('spar_inertia', SparInertia(), promotes_inputs=['L_spar', 'M_spar', 'spar_draft'], promotes_outputs=['I_spar'])

	 	self.add_subsystem('spar_cog', SparCoG(), promotes_inputs=['L_spar', 'M_spar', 'tot_M_spar', 'spar_draft'], promotes_outputs=['CoG_spar'])

	 	self.add_subsystem('tower_mass', TowerMass(), promotes_inputs=['D_tower', 'L_tower', 'wt_tower'], promotes_outputs=['M_tower'])

	 	self.add_subsystem('tower_total_mass', TowerTotalMass(), promotes_inputs=['M_tower'], promotes_outputs=['tot_M_tower'])

	 	self.add_subsystem('tower_inertia', TowerInertia(), promotes_inputs=['L_tower', 'M_tower'], promotes_outputs=['I_tower'])

	 	self.add_subsystem('tower_cog', TowerCoG(), promotes_inputs=['L_tower', 'M_tower'], promotes_outputs=['CoG_tower'])

	 	self.add_subsystem('buoyancy', Buoyancy(), promotes_inputs=['D_spar', 'L_spar'], promotes_outputs=['buoy_spar', 'CoB'])

	 	self.add_subsystem('ballast', Ballast(), promotes_inputs=['spar_draft', 'buoy_spar', 'tot_M_spar', 'L_spar', 'D_spar', 'tot_M_tower', 'M_nacelle', 'M_rotor', 'M_moor', 'rho_ball', 'wt_ball'], promotes_outputs=['M_ball', 'CoG_ball', 'L_ball'])

	 	self.add_subsystem('ball_inertia', BallInertia(), promotes_inputs=['M_ball', 'CoG_ball'], promotes_outputs=['I_ball'])

	 	self.add_subsystem('modeshape', Modeshape(), promotes_inputs=['D_spar', 'L_spar', 'wt_spar', 'M_spar', 'Z_spar', 'CoG_spar', 'D_tower', 'L_tower', 'wt_tower', 'M_tower', 'Z_tower', 'spar_draft', 'M_ball', 'CoG_ball', 'wt_ball', 'L_ball', 'M_nacelle', 'M_rotor', 'I_rotor', 'K_moor', 'M_moor', 'z_moor', 'buoy_spar', 'CoB'], promotes_outputs=['x_sparmode', 'x_towermode', 'z_sparmode', 'z_towermode'])

	 	self.add_subsystem('bending_mass', BendingMass(), promotes_inputs=['z_sparmode', 'x_sparmode', 'z_towermode', 'x_towermode', 'M_spar', 'L_spar', 'Z_spar', 'M_tower', 'L_tower', 'Z_tower', 'M_ball', 'L_ball', 'wt_ball', 'M_rotor', 'M_nacelle', 'I_rotor', 'spar_draft', 'wt_ball', 'L_ball', 'spar_draft'], promotes_outputs=['M17', 'M57', 'M77'])
	
		self.add_subsystem('bending_addedmass', BendingAddedMass(), promotes_inputs=['x_sparmode', 'z_sparmode', 'Z_spar', 'spar_draft', 'D_spar'], promotes_outputs=['A17', 'A57', 'A77'])
		
		self.add_subsystem('bending_damping', BendingDamping(), promotes_inputs=['D_tower', 'wt_tower', 'L_tower', 'Z_tower', 'x_towermode', 'z_towermode'], promotes_outputs=['B_struct_77'])
		
		self.add_subsystem('bending_stiffness', BendingStiffness(), promotes_inputs=['z_sparmode', 'x_sparmode', 'z_towermode', 'x_towermode', 'D_tower', 'wt_tower', 'M_tower', 'L_tower', 'Z_tower', 'K_moor', 'M_moor', 'z_moor', 'buoy_spar', 'CoB', 'M_spar', 'CoG_spar', 'M_ball', 'CoG_ball', 'M_rotor', 'M_nacelle'], promotes_outputs=['K17', 'K57', 'K77'])

		self.add_subsystem('aero_damping', AeroDamping(), promotes_inputs=['Z_tower', 'CoG_rotor', 'dthrust_dv', 'dmoment_dv', 'x_towermode', 'z_towermode'], promotes_outputs=['B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77'])

	 	self.add_subsystem('global_mass', GlobalMass(), promotes_inputs=['tot_M_spar', 'M_ball', 'tot_M_tower', 'M_nacelle', 'M_rotor', 'CoG_spar', 'CoG_ball', 'CoG_tower', 'CoG_nacelle', 'CoG_rotor', 'I_spar', 'I_ball', 'I_tower', 'I_rotor', 'M17', 'M57', 'M77'], promotes_outputs=['M_global'])

	 	self.add_subsystem('global_addedmass', GlobalAddedMass(), promotes_inputs=['A11', 'A15', 'A17', 'A55', 'A57', 'A77'], promotes_outputs=['A_global'])

	 	self.add_subsystem('global_damping', GlobalDamping(), promotes_inputs=['B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77'], promotes_outputs=['B_global'])

	 	self.add_subsystem('global_stiffness', GlobalStiffness(), promotes_inputs=['D_spar', 'tot_M_spar', 'tot_M_tower', 'M_nacelle', 'M_rotor', 'M_ball', 'CoG_spar', 'CoG_tower', 'CoG_nacelle', 'CoG_rotor', 'CoG_ball', 'M_moor', 'K_moor', 'K17', 'K57', 'K77', 'buoy_spar', 'CoB'], promotes_outputs=['K_global'])

	 	self.add_subsystem('wave_loads', WaveLoads(), promotes_inputs=['D_spar', 'L_spar', 'omega_wave', 'water_depth', 'z_sparmode', 'x_sparmode'], promotes_outputs=['Re_wave_forces', 'Im_wave_forces'])

	 	#self.add_subsystem('RAO_wave', RAO(), promotes_inputs=['M_global', 'A_global', 'B_global', 'K_global', 'omega_wave', 'Re_wave_forces', 'Im_wave_forces'], promotes_outputs=['Re_RAO', 'Im_RAO'])