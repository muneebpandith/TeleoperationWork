import time
import rospy
import numpy as np
from sensor_msgs.msg import JointState
import threading
rospy.init_node('allegro_try_new')


"""
/gazebo/link_states

gazebo/link_states
/gazebo/model_states
/gazebo/parameter_descriptions
/gazebo/parameter_updates
/gazebo/performance_metrics
/gazebo/set_link_state
/gazebo/set_model_state
"""

pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
rate = rospy.Rate(10)
qdes = JointState()
qdes.name=['joint_0.0','joint_1.0','joint_2.0','joint_3.0','joint_4.0','joint_5.0','joint_6.0','joint_7.0','joint_8.0','joint_9.0','joint_10.0','joint_11.0', 'joint_12.0','joint_13.0','joint_14.0','joint_15.0' ]
qdes.position = [0 for i in range(20)]	
#qdes.name = ['ahand_joint_0','ahand_joint_1', 'ahand_joint_2', 'ahand_joint_3', 'ahand_joint_4', 'ahand_joint_5',  'ahand_joint_6', 'ahand_joint_7', 'ahand_joint_8', 'ahand_joint_9', 'ahand_joint_10', 'ahand_joint_11', 'ahand_joint_12', 'ahand_joint_13', 'ahand_joint_14', 'ahand_joint_15']
try:
	choice = 1
	while choice < 6:
		choice = int(input('Enter Choice: 0. HOME 1. ROCK 2. PAPER 3. SCISSORS 4. MANUAL 5.CUBE_HOLDING_POS 6. EXIT:::: '))
		if choice == 0:
			qdes.position = [0 for i in range(16)]		

		
		elif choice == 2:
			qdes.position = [0 for i in range(16)]
		
		elif choice == 3: 
			qdes.position = [0 for i in range(16)]
			qdes.position[8] = 0*np.pi/180  #finger_pinky sideways_revolute aka joint_8.0 (NOT REQUIRED)
			qdes.position[9] = 145*np.pi/180 #finger_pinky revolute aka joint_9.0
			qdes.position[10] = 145*np.pi/180 #finger_pinky revolute aka joint_10.0
			qdes.position[11] = 145*np.pi/180 #finger_pinky revolute aka joint_11.0
			

			qdes.position[12] = 30*np.pi/180  #thumb revolute aka joint_12.0
			qdes.position[13] = 180*np.pi/180  #finger_index sideways_revolute aka joint_0.0 (NOT REQUIRED)
			qdes.position[14] = 145*np.pi/180 #thumb revolute aka joint_14.0
			qdes.position[15] = 145*np.pi/180 #thumb revolute aka joint_15.0
			
		elif choice == 1:
			qdes.position[0] = 0*np.pi/180 #finger_index sideways revolute aka joint_0.0 (NOT REQUIRED)
			qdes.position[1] = 145*np.pi/180 #finger_index revolute aka joint_1.0
			qdes.position[2] = 145*np.pi/180 #finger_index revolute aka joint_2.0
			qdes.position[3] = 145*np.pi/180 #finger_index revolute aka joint_3.0
			
			qdes.position[4] = 0*np.pi/180  #finger_middle sideways_revolute aka joint_4.0 (NOT REQUIRED)
			qdes.position[5] = 145*np.pi/180 #finger_middle revolute aka joint_5.0
			qdes.position[6] = 145*np.pi/180 #finger_middle revolute aka joint_6.0
			qdes.position[7] = 145*np.pi/180 #finger_middle revolute aka joint_7.0
			
			qdes.position[8] = 0*np.pi/180  #finger_pinky sideways_revolute aka joint_8.0 (NOT REQUIRED)
			qdes.position[9] = 145*np.pi/180 #finger_pinky revolute aka joint_9.0
			qdes.position[10] = 145*np.pi/180 #finger_pinky revolute aka joint_10.0
			qdes.position[11] = 145*np.pi/180 #finger_pinky revolute aka joint_11.0
			
			

			qdes.position[12] = 30*np.pi/180  #thumb revolute aka joint_12.0
			qdes.position[13] = 180*np.pi/180  #finger_index sideways_revolute aka joint_0.0 (NOT REQUIRED)
			qdes.position[14] = 145*np.pi/180 #thumb revolute aka joint_14.0
			qdes.position[15] = 145*np.pi/180 #thumb revolute aka joint_15.0
			
		elif choice == 5:
			qdes.position = [0.09895952643811425, 0.6854171329610701, 0.7271570460226513, 1.1372475342255288, -0.057032108983673044, 0.7441264917442691, 0.519628437640359, 1.3211759917961334, 0.036433434000255546, -0.20450542223751048, 0.6632740936226852, 0.7857053505148868, 1.4845574309720937, 0.22270548112331584, -0.29327417871147854, 1.6768425067218493]

		elif choice == 4:
			qdes.position = [0.6, 0.6854171329610701, 0.7271570460226513, 1.1372475342255288, -0.057032108983673044, 0.7441264917442691, 0.519628437640359, 1.3211759917961334, 0.036433434000255546, -0.20450542223751048, 0.6632740936226852, 0.7857053505148868, 1.4845574309720937, 0.22270548112331584, -0.29327417871147854, 1.6768425067218493]

		elif choice == 6: 
			print('Exitting')
			break
		try:
			pub.publish(qdes)
		except:
			print('noway')

except:
	print('error')

