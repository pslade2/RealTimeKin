import numpy as np
import ahrs
import time
from ahrs.common.orientation import q2R

# Function to write data from line_ind to a .sto file
def quat2sto_single(sensor_data, header_text, file_dir, t_step, rate, sensor_ind_list = [4,4,2,3,0,1]):
    num_sensors, _ = sensor_data.shape
    with open(file_dir, 'w') as f:
        # initial information to write to file
        f.write("DataRate={}\n".format(rate))
        f.write("DataType=Quaternion\n")
        f.write("version=3\n")
        f.write("OpenSimVersion=4.2\n")
        f.write("endheader\n")
        f.write(header_text)
        f.write("{}".format(t_step))
        for j in sensor_ind_list:
            f.write("\t{},{},{},{}".format(sensor_data[j,0],sensor_data[j,1],sensor_data[j,2],sensor_data[j,3]))
        f.write("\n")

# Function to read sto file to load fake real time data in numpy array
def sto2quat(file_dir, lines = 3, offset = 6, num_sensors=8):
    sensor_data = np.zeros((num_sensors,lines,4))
    times = np.zeros(lines)
    with open(file_dir, 'r') as f:
        for i in range(lines+offset):
            line = f.readline()
            if i >= offset:
                row_ind = i - offset
                sect_tab = line.split('\t')
                for j, sect in enumerate(sect_tab):
                    if j == 0:
                        times[row_ind] = float(sect)
                    else:
                        subsect = sect.split(',')
                        for k,val in enumerate(subsect):
                            sensor_data[j-1,row_ind,k] = float(val)
    return times, sensor_data

def compute_quat(all_data, len_sensor_list, quat_cal_offset, rot_inds, num_sensors=5, t_offset=0, signals_per_sensor=6, beta=0.2*np.ones(5), rate=60.0, verbose=False, beta_init=0.4):
    d2g = ahrs.common.DEG2RAD   # Constant to convert degrees to radians
    Qi = np.zeros((num_sensors,quat_cal_offset,4))
    Q = np.zeros((num_sensors,all_data.shape[0]-quat_cal_offset,4))
    Qi[:,0,:] = np.array([0.7071, 0.7071, 0.0, 0.0])

    z_neg_90 = np.array([[0,1.0,0],[-1.,0,0],[0,0,1.0]])
    y_180 = np.array([[-1.0,0,0],[0,1.0,0],[0,0,-1.0]])
    z_180 = np.array([[-1.0,0,0],[0,-1.0,0],[0,0,1.0]])
    y_neg_90 = np.array([[0,0,-1.0],[0,1.0,0],[1.0,0,0]])
    y_pos_90 = np.array([[0,0,1.0],[0,1.0,0],[-1.0,0,0]])
    ankle_offset = -100.*d2g 
    x_pos_ankle = np.array([[1.0,0,0],[0,np.cos(ankle_offset),-np.sin(ankle_offset)],[0,np.sin(ankle_offset),np.cos(ankle_offset)]])
    hip_rot = np.matmul(y_neg_90,z_180)
    foot_rot = np.matmul(x_pos_ankle, hip_rot) 
    r_leg_rot = z_neg_90 
    l_leg_rot = np.matmul(z_neg_90,y_180)
    rot_mats = np.zeros((len_sensor_list,3,3))
    for i in range(len_sensor_list): # define rotation type
        if rot_inds[i] == 0: # hip, torso, head
            rot_mats[i,:,:] = hip_rot
        elif rot_inds[i] == 1: # left side
            rot_mats[i,:,:] = l_leg_rot
        elif rot_inds[i] == 2: # right side
            rot_mats[i,:,:] = r_leg_rot
        elif rot_inds[i] == 3: # foot
            rot_mats[i,:,:] = foot_rot

    for i in range(len_sensor_list): # processing quaternions one sensor at a time
        s_off = i*signals_per_sensor
        accel = np.matmul(all_data[:,s_off+t_offset:s_off+t_offset+3],rot_mats[i,:,:])
        gyro = np.matmul(all_data[:,s_off+t_offset+3:s_off+t_offset+6],rot_mats[i,:,:])
        #mag = np.matmul(all_data[:,s_off+t_offset+6:s_off+t_offset+9],rot_mats[i,:,:])
        
        # calibrating the initial quaternion with a large beta_init value
        madgwick_i = ahrs.filters.Mahony(frequency=float(rate))
        for t in range(1, quat_cal_offset):
            Qi[i,t,:] = madgwick_i.updateIMU(Qi[i,t-1,:], gyro[0,:], accel[0,:])
    quat_ang = np.zeros(num_sensors)
    for i in range(len_sensor_list):
        quat_ang[i],_ = orientMat(q2R(np.array(Qi[i,-1,:]))) # multiply rot_mats by the gyro correction angle to correct to same heading as pelvis? Then
    mean_ang = np.mean(quat_ang)
    return Qi[:,-1,:], mean_ang, rot_mats

def rotateY(th, R):
    Ry = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    newR = np.matmul(R,Ry)
    return newR

def orientMat(R, num_angles=100, R_ref=np.array([[1.,0.,0. ],[0.,0.,-1.],[0.,1.,0.]])):
    angles = np.linspace(-np.pi,np.pi,num_angles)
    best_angle = -np.pi
    best_distance = np.inf
    for a in angles:
        R_mat = rotateY(a,R)
        norm = np.linalg.norm(R_ref - R_mat)
        if norm < best_distance:
            best_distance = norm
            best_angle = a
    return best_angle, best_distance
