import rospy
import numpy as np
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import Robot
import math
import time
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from scipy.linalg import logm as logm
import csv
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as sk
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
rospy.init_node('qaim_jacob_multiply', anonymous=True)

global default_hold_conf
global pub_to_ah


#filename= 'trace_theta'+str(time.time())+'.csv'
#Saving the traces of data for analysis
filename2 = '../DATASET/TRANSORMER_DATA/training_data_for_transformer'+str(time.time())+'.csv'

def save_to_csv(FILENAME, TRACE_CSV, type_open='a'):    
	with open(FILENAME,type_open,newline="") as trace_file:
		writer = csv.writer(trace_file, )
		writer.writerow(TRACE_CSV)


class LSTMNET(nn.Module):
    def __init__(self, batch_size, input_len, output_len, lstm_units = 4, num_layers=1):
        super(LSTMNET, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_layers = num_layers
        self.batch_size = batch_size
        #input_size = no of features = 1
        #hidden_size = no of lstm units in the layer
        #num_layers = no of lstm layers
        self.lstm_units = lstm_units
        self.lstm1 = nn.LSTM(input_size= 1, hidden_size= lstm_units, num_layers=num_layers,batch_first=True, dropout=0.6)
        
        self.linear0 = nn.Linear(in_features= 20, out_features=10)
        
        self.linear1 = nn.Linear(in_features= lstm_units, out_features=10)
        self.linear2 = nn.Linear(in_features= 10, out_features=10)
        self.ll = nn.Linear(in_features= 10, out_features=output_len)
        self.hidden = (torch.zeros(1*self.num_layers, self.batch_size, self.lstm_units).double(), torch.zeros(1*self.num_layers, self.batch_size, self.lstm_units).double())
        #print(self.hidden[0].device)
        #print(self.hidden.shape)
        #self.hidden[0]= self.hidden[0].to(device)
        #self.hidden[1] = self.hidden[1].to(device)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def reset_hidden_states(self,bs):
        self.hidden = (torch.zeros(1*self.num_layers, bs, self.lstm_units).double(), torch.zeros(1*self.num_layers, bs, self.lstm_units).double())
        
    def forward(self,x):
        #print(x.shape)
        #print(x)
        #print(x.unsqueeze(-1).shape)
        #print(self.hidden.shape)
        #print(self.hidden)
        lstm_out, (h,c) = self.lstm1(x.unsqueeze(-1), self.hidden)
        self.hidden= (h.detach(),c.detach())
        #c.detach_()
        #h.detach_()
        #self.hidden = (h.detach(), c.detach())
        #print(ht.shape)
        #ht=ht.to(device)
        #ct=ct.to(device)
        
        lstm_out = lstm_out[:,-1,:]
        #print(ht.shape)
        #either lstm_out goes to next or ht goes
        #lstm_out= h.detach()[-1]
        lin1_out = self.linear1(lstm_out)
        #Add RELU
        #lin0_out = F.relu(self.linear0(x))
        ll_out = self.ll(lin1_out)
        #x = self.linear0(x)
        #print(x.shape)
        
        #x = self.linear2(x)
        #Add RELU
        return ll_out
input_cardinality = 20
output_cardinality = 4

model_ckp_path = "../MODEL_CHECKPOINTS/model_lookahead"+str(output_cardinality)+".pth"
scaler_filename = "../SCALER_DUMPS/min_max_scaler_lookahead"+str(output_cardinality)+".save"
batch_size = 1
predictor = LSTMNET(batch_size=batch_size,input_len=input_cardinality,output_len=output_cardinality)
predictor.load_state_dict(torch.load(model_ckp_path))
#net2.eval()
predictor.reset_hidden_states(bs=1)
scaler = joblib.load(scaler_filename) 

predictor.eval() 
predictor = predictor.double()



header_dataa = ['J'+str(i) for i in range(16)]
header_dataa.append('ThetaXHG')
header_dataa.append('ThetaYHG')
header_dataa.append('ThetaZHG')
save_to_csv(filename2, header_dataa, 'w')


default_hold_conf = [-0.03173544473219131, 0.6567962627624016, 0.9265637905098385, 0.5383870490896738, -0.03547774473318004, 0.7488147923462423, 0.673867398677253, 0.9823142433473055, 0.008727332299797975, -0.11442395562499424, 0.3272287805964458, 0.333846920707705, 1.5223462924197833, 0.09608335774758818, -0.2929165280825893, 1.5039146258794902]
#default_hold_conf = rospy.wait_for_message('/allegroHand_0/joint_states', JointState)
#use default_hold_conf.position

pub_to_ah = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
rospy.Rate(10)

class qaim_allegro:
	def __init__(self):
		#rospy.init_node('qaim_work')
		self.joint_set = []
		urdf_file = open('/home/toor/Desktop/catkin_ism/src/allegro-hand-ros/allegro_hand_description/allegro_hand_description_right.urdf', 'r')  #get in the urdf file
		self.robot = Robot.from_xml_string(urdf_file.read())
		urdf_file.close()
		self.tree = kdl_tree_from_urdf_model(self.robot)

		#kinematics
		
		self.kdl_kin1 = KDLKinematics(self.robot,'palm_link', 'link_3.0_tip')
		self.kdl_kin2 = KDLKinematics(self.robot,'palm_link', 'link_7.0_tip')
		self.kdl_kin3 = KDLKinematics(self.robot,'palm_link', 'link_11.0_tip')
		self.kdl_kin4 = KDLKinematics(self.robot,'palm_link', 'link_15.0_tip')
		
global allegro_hand
allegro_hand = qaim_allegro()

def mapper_angle(data_HG):
	#This function maps HG angles (split, bend) to angles in Radians, example split = [0,1] => RAdians [0,180]
	q = JointState()
	q_return = np.zeros(11)

	q = data_HG

	for i in range(11):
		if i == 0:
			continue
		elif i == 1: #thumb_split
			q_return[i] = (-262.7231*q.position[i] + 200.63926) /(180/np.pi)
		elif i == 2: #thumb_bend
			q_return[i] = (117.20154*q.position[i] - 0.58601)/(180/np.pi)
		#REST BEND
		elif i == 4 or i == 6 or i == 8 or i == 10:
			q_return[i] = (2.644 * q.position[i] - 0.4375)
		#REST SPLIT
		elif i == 3 or i == 5 or i == 7 or i ==9:
			q_return[i] = (100* q.position[i])/(180/np.pi)
	return q_return

def joint_rate(q1,q2,t1,t2):
	q_rate = np.array((np.array(q2)-np.array(q1)))/(t2-t1)
	return q_rate 

def get_haptic_glove_data():
	data_HG = JointState()
	data_HG = rospy.wait_for_message('/glove_out', JointState)
	return data_HG

def get_haptic_glove_data_dummy():
	data_HG = JointState()
	data_HG.position = [0 for i in range(11)]
	return data_HG

def get_allegro_hand_data():
	data_AH = JointState()
	data_AH = rospy.wait_for_message('/allegroHand_0/joint_states', JointState)
	return data_AH

def get_allegro_hand_data_dummy():
	data_AH = JointState()
	data_AH.position =  [0.01902325540010691, 0.8620307161294611, 0.3939724544336821, 1.5135765027138792, -0.1928997470681366, 0.7439964307241076, 0.7449952271027224, 1.2314336722179493, 0.20788792510998383, -0.04702816242730262, -0.2845939771330153, 0.5118857671793973, 1.1421136294639596, 1.191262942722061, 0.1039615018338904, 1.742074945210235]
	return data_AH


def get_vector():
	data  = Floats()
	data = rospy.wait_for_message('/vector', numpy_msg(Floats))
	#data.data= [0,0] 
	#SCALE = actual_distance/(topLeft-topRight), top left is top left of ARUCO! only the black part.
	# Lets assume for now, x and y have scaled values.
	return data

def get_angle_from_camera():
	data  = Floats()
	data = rospy.wait_for_message('/current_angle', numpy_msg(Floats))
	return data

def get_angle_from_camera_dummy():
	angle  = Floats()
	angle.data = [0]
	return angle


def get_vector_dummy():
	vector  = Floats()
	scale = 0.04/28
	vector.data = [0.02769230678677559, -0.035384614020586014, -0.02769230678677559, -0.035384614020586014, 0.0, 0.03076923079788685]
	return vector

def pose(delta_theta):
	
	theta = np.linalg.norm(delta_theta)
	E = np.array([[0, -delta_theta[2,0], delta_theta[1,0]], [delta_theta[2,0], 0, -delta_theta[0,0]], [-delta_theta[1,0], delta_theta[0,0], 0]])
	laplace = 0.0000001
	E = (1/(theta+laplace))*E
	I = np.identity(3)
	R = I + E*math.sin(theta) + np.matmul(E,E)*(1 - math.cos(theta))
	return R
"""
def set_cube_holding_pose_sim():
	pub = rospy.Publisher('/glove_out_sim',JointState, queue_size=10)
	for i in range(20):
		
		qdes = JointState()
		qdes.name=['joint_0.0','joint_1.0','joint_2.0','joint_3.0','joint_4.0','joint_5.0','joint_6.0','joint_7.0','joint_8.0','joint_9.0','joint_10.0','joint_11.0', 'joint_12.0','joint_13.0','joint_14.0','joint_15.0' ]
		qdes.position = [(i%2) for j in range(53)]	
		pub.publish(qdes)
		print(qdes)
		time.sleep(1)

def set_cube_holding_pose():
	A = JointState()
	A.name =['joint_0.0','joint_1.0','joint_2.0','joint_3.0','joint_4.0','joint_5.0','joint_6.0','joint_7.0','joint_8.0','joint_9.0','joint_10.0','joint_11.0', 'joint_12.0','joint_13.0','joint_14.0','joint_15.0' ]
	
	for i in range(1000):
		try:
			A.position= [0.02924683788043305, 0.3894144907313105, 1.290775091530005, 0.8171478783694223, -0.013263785176304888, 0.49311782150858346, 1.1976323206297346, 0.8956232893613544, -0.10424139662387323, -0.26750957104723605, 0.08616930055726134, 0.8199506468110072, 1.378570268659443, 0.48660390023527855, -0.21675098438696405, 1.743271653674864,0,0,0,0]
			#data_AH.position =[1 for i in range(16)]
			pub_to_ah.publish(A)
		
		except Exception as e:
			print(e)
		rospy.spin()
"""

def set_initial_params(debug):
	PHI = np.zeros(6)
	Theta0 = np.array([0,0,0]).reshape(3,)

	alpha = 0.6144285978268563
	beta = 0.4739877790539716
	gamma = 0.13415630605
	DATA_AH = JointState()
	DATA_AH.name =['joint_0.0','joint_1.0','joint_2.0','joint_3.0','joint_4.0','joint_5.0','joint_6.0','joint_7.0','joint_8.0','joint_9.0','joint_10.0','joint_11.0', 'joint_12.0','joint_13.0','joint_14.0','joint_15.0' ]
	DATA_AH.position = [0 for i in range(16)]
	DATA_AH_2 = JointState()
	DATA_AH_2.name =['joint_0.0','joint_1.0','joint_2.0','joint_3.0','joint_4.0','joint_5.0','joint_6.0','joint_7.0','joint_8.0','joint_9.0','joint_10.0','joint_11.0', 'joint_12.0','joint_13.0','joint_14.0','joint_15.0' ]
	DATA_AH_2.position = np.array([0 for i in range(16)])
	DATA_HG = JointState()
	return PHI, Theta0, alpha, beta, gamma



def get_desired_theta(Q1, Q2, t1, t2, PHI, Theta0, alpha, beta, gamma, data_AH=JointState(), start=False, debug=False):
	q_rate = joint_rate(Q1,Q2,t1,t2)
	#print(q_rate)

	#0: thumb_rotate, 1:thumb_split, 2:thumb_bend, 3:index_split, index_bend, middle_split, middle_bend, ring_splitring_bend, last_split, last_bend
	for j in range(6):
		PHI[j] = Q2[j+1]
			#PHI[0] =THUMB split
			#PHI[1] =THUMB bend
			#PHI[2] = INDEX split ==DONE
			#PHI[3] = INDEX bend
			#PHI[4] = MIDDLE split
			#PHI[5] = MIDDLE BEND
			
	w1_split = q_rate[3]*np.array([[1],[0],[0]])  # index finger spliT
	w1_bend = q_rate[4]*np.matmul(np.array([[1,0,0],[0,math.cos(PHI[2]-alpha),math.sin(PHI[2] -alpha)],[0,-math.sin(PHI[2] -alpha),math.cos(PHI[2]-alpha)]]),np.array([[0],[1],[0]])) #index finer bend
	w1 = (w1_split + w1_bend).reshape(3,)

	w2_split = q_rate[5]*np.array([[1],[0],[0]]) # middle finger spliT
	w2_bend = q_rate[6]*np.matmul(np.array([[1,0,0],[0,math.cos(PHI[4]-beta),math.sin(PHI[4]-beta)],[0,-math.sin(PHI[4]-beta),math.cos(PHI[4]-beta)]]),np.array([[0],[1],[0]]))
	w2 = (w2_split + w2_bend).reshape(3,)
	#[index_x_acc_camera,index_y_acc_camera,middle_x_acc_camera,middle_y_acc_camera,thumb_x_acc_camera,thunb_y_acc_camera]

	w3_split = q_rate[1]*np.array([[-0.3472],[0.7577],[-0.5525]])  # thumb split
	thumb_split_pose = pose(-(gamma-PHI[0])*np.array([[-0.3472],[0.7577],[-0.5525]]))
	w3_bend = q_rate[2]*np.matmul(thumb_split_pose,[[-0.22],[-0.6897],[-0.6897]])  # thumb bend 
	w3 = (w3_split + w3_bend).reshape(3,)

	#Get data from robotic hand
	if start == True:
		if debug == True:
			DATA_AH = get_allegro_hand_data_dummy()
			DATA_AH_2 = get_allegro_hand_data()
		else:
			DATA_AH = get_allegro_hand_data()
			DATA_AH_2 = get_allegro_hand_data() #coming from hand
	else:
		DATA_AH = data_AH
		DATA_AH_2 = get_allegro_hand_data() #coming from hand
	
	#print(DATA_AH.position)
	#print(DATA_AH_2.position)
	#print(DATA_AH.position)
	r1 = allegro_hand.kdl_kin1.forward(np.array(DATA_AH_2.position)[0:4])   #SLICE PERFINGER FOR INDEX MIDDLE THUMB, WILL BE A 4X4 MATRIX, SLICE ITS LAST COLUMN EXCEPT UNITARY ELEMENT IN THE END OF THE COLUMN

	SUBTRACTION_FACTOR_INDEX = np.array([0, 0.0435, -0.001542]).reshape(3,) #taken from URDF file for index finger
	SUBTRACTION_FACTOR_MIDDLE = np.array([0, 0, 0.0007]).reshape(3,)   #taken from URDF file for middle finger
	SUBTRACTION_FACTOR_THUMB = np.array([-0.0182, 0.019333, -0.045987]).reshape(3,)  #taken from URDF file for thumb
	
	r1_position = np.array([-1,-1,-1])*(np.array(r1[0:3,3]).reshape(3,)-SUBTRACTION_FACTOR_INDEX)
	
	#So r1 must be 3x1
	r2 = allegro_hand.kdl_kin2.forward(np.array(DATA_AH_2.position)[4:8])   #SLICE PERFINGER FOR INDEX MIDDLE THUMB, WILL BE A 4X4 MATRIX, SLICE ITS LAST COLUMN EXCEPT UNITARY ELEMENT IN THE END OF THE COLUMN
	
	r2_position = np.array([-1,-1,-1])*(np.array(r2[0:3,3]).reshape(3,)-SUBTRACTION_FACTOR_MIDDLE)
	#So r2 must be 3x1
	r3 = allegro_hand.kdl_kin4.forward(np.array(DATA_AH_2.position)[12:16])   #SLICE PERFINGER FOR INDEX MIDDLE THUMB, WILL BE A 4X4 MATRIX, SLICE ITS LAST COLUMN EXCEPT UNITARY ELEMENT IN THE END OF THE COLUMN
	r3_position = np.array([-1,-1,-1])*(np.array(r3[0:3,3]).reshape(3,)-SUBTRACTION_FACTOR_THUMB)
	#So r3 must be 3x1


	#print(w1.shape, r1_position.shape)
	#print(w1, r1_position)
	#print(w2, r2_position)
	#print(w3, r3_position)

	v1 = np.cross(w1,r1_position)  # 3x1 => 3x3 or  using np.corss 
	v2 = np.cross(w2,r2_position)
	v3 = np.cross(w3,r3_position)
	v = np.vstack((v1,v2,v3)).reshape(9,)
	#print(v)
	#v is 9x1 consisting vertical stacking of v1,v2 and v3
	#r%0d  = vector [x,y] from cernter of three fingers

	if debug == True:
		vector_data = get_vector_dummy()
	else:
		vector_data = get_vector()

	# vector_data.data = [index_x_acc_camera,index_y_acc_camera,middle_x_acc_camera,middle_y_acc_camera,thumb_x_acc_camera,thunb_y_acc_camera]

	#RHS IS CAMERA x,y, LHS is HAND/GLOVE REF X,Y
	#y' = x
	#z' = y
	#x= 0
		
	Z1 = vector_data.data[1]
	Y1 = -vector_data.data[0]
	X1 = 0

	Z2 = vector_data.data[3]
	Y2 = -vector_data.data[2]
	X2 = 0

	Z3 = vector_data.data[5]
	Y3 = -vector_data.data[4]
	X3 = 0

	r01 = np.array([[0,-Z1,Y1],[Z1,0,-X1],[-Y1,X1,0]])
	r02 = np.array([[0,-Z2,Y2],[Z2,0,-X2],[-Y2,X2,0]])
	r03 = np.array([[0,-Z3,Y3],[Z3,0,-X3],[-Y3,X3,0]])

	#convert r%D%D to 3x3 for preparation to cross product
	#then stack r%D%D vertically TO MAKE 9x3 matirxdebug)
		
	#print(np.linalg.pinv(np.vstack((-r01,-r02,-r03))))
	w0 = np.matmul(np.linalg.pinv(np.vstack((-r01,-r02,-r03))),v).reshape(3,)
	#print(w0)

	#pinv will give 3x9 matrix
	#w0 will contain 3x1 elemetns
	#integrate w0 over t2-t1 this will give us angular displcaement
	Angular_Displacement =  w0*(t2-t1)  #angular diaplcaement
	#print(t2-t1)
	Theta0 = Theta0 + Angular_Displacement  #np.array([1,0,0]).reshape(3,)
	
	if start == False:
		return np.array([1,0,0])*Theta0, r01, r02, r03
	else:
		return np.array([1,0,0])*Theta0, r01, r02, r03, DATA_AH 
	
	#integrate ML to generate futre Theta0


def control_AH(Theta0, r01, r02, r03, t1, t2, DATA_AH, 	const1, debug):
	#print("++++++++++++++++++++++++===")
	print(Theta0)
	desired_pose = pose(Theta0.reshape(3,1))    #gives 3x3 matrix
	if debug == True:
		angle_from_camera = get_angle_from_camera_dummy()
	else:
		angle_from_camera = get_angle_from_camera()
		
	curr_pose = pose(np.array([angle_from_camera.data[0],0,0]).reshape(3,1))          #(as 3x1))   #gives 3x3 matirx
	
	#print(curr_pose)
	#print(desired_pose)

	#print(np.matmul(curr_pose.transpose(), desired_pose).shape)
	#print(laplace+ curr_pose.transpose())
	desired_angular_velocity_object = logm(np.matmul(curr_pose.transpose(), desired_pose))  #gives a 3x3 matrix which can be changed into 3x1 matrix of angular velocity
	#print(desired_angular_velocity_object)
	desired_angular_velocity_object_3x1 = np.array([[desired_angular_velocity_object[2,1]],[desired_angular_velocity_object[0,2]],[desired_angular_velocity_object[1,0]]])
	#print(desired_angular_velocity_object_3x1)

	twist_index = np.matmul(r01,desired_angular_velocity_object_3x1)        #desired velocities of tips of RH fingers
	twist_middle = np.matmul(r02,desired_angular_velocity_object_3x1)
	twist_thumb = np.matmul(r03,desired_angular_velocity_object_3x1)

	#twist_RH = np.vstack((twist_index,twist_middle,twist_thumb))

	jacobian_index = allegro_hand.kdl_kin1.jacobian(np.array(DATA_AH.position)[0:4])
	jacobian_middle = allegro_hand.kdl_kin2.jacobian(np.array(DATA_AH.position)[4:8])
	jacobian_thumb = allegro_hand.kdl_kin4.jacobian(np.array(DATA_AH.position)[12:16])
	j1 = jacobian_index[0:3,:]
	j2 = jacobian_middle[0:3,:]
	j3 = jacobian_thumb[0:3,:]

	#J = np.vstack((j1,j2,j3))

	desired_joint_rate_index_RH = np.matmul(np.linalg.pinv(j1),twist_index)  #desired joint rates of corresponding RH fingers
	desired_joint_rate_middle_RH = np.matmul(np.linalg.pinv(j2),twist_middle)
	desired_joint_rate_thumb_RH = np.matmul(np.linalg.pinv(j3),twist_thumb)
	#print(desired_joint_rate_index_RH.shape, desired_joint_rate_middle_RH.shape, desired_joint_rate_thumb_RH.shape)
	desired_joint_rate_ring_RH = np.array([[0],[0],[0],[0]])

	#print(desired_joint_rate_index_RH,desired_joint_rate_middle_RH,desired_joint_rate_ring_RH ,desired_joint_rate_thumb_RH)
	desired_joint_rate_RH = np.ravel(np.vstack((desired_joint_rate_index_RH,desired_joint_rate_middle_RH,desired_joint_rate_ring_RH,desired_joint_rate_thumb_RH)))
	#print(desired_joint_rate_RH.shape)


	#print(t2-t1)
	#print(DATA_AH.position)
	#print(desired_joint_rate_RH)
	#print(const1*(t2-t1))
	#DATA_AH2 = JointState()
	#print("***********************")
	#print(DATA_AH.position)
	#print(const1*(t2-t1)*desired_joint_rate_RH)
	
	"""
	print("****")
	print(np.array(DATA_AH.position))
	print(const1)
	print(t2-t1)
	print(desired_joint_rate_RH)
	print(15*const1*(t2-t1)*desired_joint_rate_RH)
	"""
	POSITION_ARRAY = np.array(DATA_AH.position) + const1*(t2-t1)*desired_joint_rate_RH
	#print(POSITION_ARRAY)
	DATA_AH.position = [POSITION_ARRAY[i] for i in range(16)]
	
	#print(DATA_AH.position)
	#print("====")

	#DATA_AH.effort = [1 for i in range(16)]
	#DATA_AH2.position = np.array(DATA_AH.position) + const1*(t2-t1)*desired_joint_rate_RH.reshape(-1,16)
	#DATA_AH2.position= [1 for i  in range(16)]
	#print(DATA_AH2.position)
	#print(type(pub_to_ah))
	save_to_csv(filename2, list(DATA_AH.position) + list([Theta0[0], Theta0[1], Theta0[2]]) ,'a')
	return DATA_AH
	#print(DATA_AH)
	#time.sleep(0.01)   #this i addeD



def main(debug):
	#RUN cube holding pose.py prior to this script
	#1. Set Data Structure of basic variables
	PHI, Theta0, alpha, beta, gamma = set_initial_params(debug)
	ALL_INPUT_TO_LSTM=list()
	#set_cube_holding_pose()

	#2. GET ANGLE MAPPED HAPTIC GLOVE DATA
	t1 = time.time()
	if debug == True:
		q1 = get_haptic_glove_data_dummy()	
	else:
		q1= get_haptic_glove_data()
	Q1 =  mapper_angle(q1)

	
	n_time = 0
	#3. LOOP FOR: GET_DESIREDTHETA, CONTROL_AH
	while True:
		Q1 =  mapper_angle(q1)
		#### time.sleep(1/240.)  COMMENTED ON 8SEP
		#while True:
		t2= time.time()
		if debug == True:
			q2 = get_haptic_glove_data_dummy()
		else:
			q2 = get_haptic_glove_data()
		Q2 = mapper_angle(q2)
		#3. GET DESIRED THETA
		if n_time==0:
			Theta0, r01, r02, r03, DATA_AH = get_desired_theta(Q1, Q2, t1, t2, PHI, Theta0, alpha, beta, gamma, JointState(), True, debug)
		else:
			Theta0, r01, r02, r03 = get_desired_theta(Q1, Q2, t1, t2, PHI, Theta0, alpha, beta, gamma, DATA_AH, False, debug)
		#print(DATA_AH.position)
		#RNN Pipeline
		ALL_INPUT_TO_LSTM.append(Theta0[0])
		if n_time>20:
			#about certain 'x' axis
			INPUT_TO_LSTM= ALL_INPUT_TO_LSTM[-input_cardinality-1:-1]
			print(len(INPUT_TO_LSTM))
			#print(INPUT_TO_LSTM)

			INPUT_TO_LSTM_scaled = scaler.transform(np.asarray(INPUT_TO_LSTM).reshape(len(INPUT_TO_LSTM),1))
			INPUT_TO_LSTM_torch_scaled = torch.tensor(INPUT_TO_LSTM_scaled.transpose(), dtype=torch.float64)
			Theta0_pred_scaled = predictor(INPUT_TO_LSTM_torch_scaled)
			Theta0_pred = scaler.inverse_transform(Theta0_pred_scaled.detach().numpy())
			Theta0[0] = Theta0_pred[0][output_cardinality-1]
		print("*********")	
		#4. CONTROL AH


		#4. CONTROL AH
		const1= 0.7
		DATA_AH = control_AH(Theta0, r01, r02, r03, t1, t2, DATA_AH, const1,debug)
		pub_to_ah.publish(DATA_AH)
		#Fprint(const1*(t2-t1))
		#time.sleep(1/10.)np.array([5, 0, 0]).reshape(3,)
		t1 = t2
		q1.position = list(q2.position).copy()
		n_time = n_time+1


if __name__=="__main__":
	main(debug=False)

#position: [0.6610000133514404, 0.6306576728820801, 0.23201754689216614, 0.6783333420753479, 0.09170305728912354, 0.5249999761581421, 0.0010000000474974513, 0.2199999988079071, 0.0010000000474974513, 1.0, 0.0429319366812706]
#position22jul: position: [0.01902325540010691, 0.8620307161294611, 0.3939724544336821, 1.5135765027138792, -0.1928997470681366, 0.7439964307241076, 0.7449952271027224, 1.2314336722179493, 0.20788792510998383, -0.04702816242730262, -0.2845939771330153, 0.5118857671793973, 1.1421136294639596, 1.191262942722061, 0.1039615018338904, 1.742074945210235]
