#!/usr/bin/python3

### This is a version of the ik_streaming.py script used to test parallelizing the cores for faster IMU estimation.
### This causes overheating and requires significant additional cooling. This is provided for experimentation but is not supported.

# Estimates kinematics from IMU data processed into quaternions using a musculoskeletal model
import opensim as osim
from opensim import Vec3
import numpy as np
from helper import quat2sto_single, sto2quat
import helper as h
import time
import os
import sys
from multiprocessing import Process, Queue
import workers # define the worker functions in this .py file
import warnings

def clear(q):
    try:
        while True:
            q.get_nowait()
    except:
        pass

warnings.filterwarnings("ignore", message="None")


# Parameters for IK solver
fake_real_time = False # True to run offline, False to record data and run online
log_temp = True
log_data = True # if true save all IK outputs, else only use those in reporter_list for easier custom coding
uncal_model = 'Rajagopal_2015.osim'
uncal_model_filename = '/home/ubuntu/RealTimeKin/' + uncal_model
model_filename = '/home/ubuntu/RealTimeKin/calibrated_' + uncal_model
fake_online_data = '/home/ubuntu/RealTimeKin/test_data.npy'
sto_filename = '/home/ubuntu/RealTimeKin/tiny_file.sto'
visualize = False
parallelize = False
rate = 20.0 # samples hz of IMUs
accuracy = 0.001 # lower is faster
constraint_var = 10.0 
sim_len = 600.0 # max time in seconds to collect data before requiring re-intialization
kin_store_size = sim_len + 10.0 
init_time = 3.0 # seconds of data to initialize from
if log_temp:
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()

# Initialize the quaternions
sim_steps = int(sim_len*rate)
signals_per_sensor = 6
file_cnt = 0
save_dir = '/home/ubuntu/RealTimeKin/recordings/test_folder/'
save_file = '/recording_'
ts_file = '/timestamp_'
script_live = True
try: # create save dir or count number of files so I don't save over some
    os.makedirs(save_dir)
except:
    f = os.listdir(save_dir)
    for s in f:
        if 'rec' in s:
            file_cnt += 1

q = Queue() # queue for IMU messages
b = Queue() # queue for button messages
imuProc = Process(target=workers.readIMU, args=(q, b, fake_real_time, init_time, signals_per_sensor,))
imuProc.start() # spawning IMU process
sensor_ind_list, rate, header_text, parallelize = b.get()
dt = 1/rate

while(script_live):
    while(q.qsize()>0): # clearing the queues that may have old messages
        q.get()
    while(b.qsize()>0):
        b.get()
    print("Ready to initialize...")
    init_time, Qi, head_err = q.get()
    # calibrate model and save
    quat2sto_single(Qi, header_text, sto_filename, 0., rate, sensor_ind_list)
    visualize_init = False
    sensor_to_opensim_rotations = Vec3(-np.pi/2,head_err,0)
    imuPlacer = osim.IMUPlacer();
    imuPlacer.set_model_file(uncal_model_filename);
    imuPlacer.set_orientation_file_for_calibration(sto_filename);
    imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations);
    imuPlacer.run(visualize_init);
    model = imuPlacer.getCalibratedModel();
    model.printToXML(model_filename)

    # Initialize model
    reporter_list = ['hip_flexion_r','knee_angle_r','hip_flexion_l','knee_angle_l'] # 'ankle_angle_r','ankle_angle_l'
    rt_samples = int(kin_store_size*rate)
    kin_mat = np.zeros((rt_samples, len(reporter_list)))
    time_vec = np.zeros((rt_samples,2))
    coordinates = model.getCoordinateSet()
    ikReporter = osim.TableReporter()
    ikReporter.setName('ik_reporter')
    for coord in coordinates:
        if log_data:
            ikReporter.addToReport(coord.getOutput('value'),coord.getName())
        elif coord.getName() in reporter_list:
            ikReporter.addToReport(coord.getOutput('value'),coord.getName())
    model.addComponent(ikReporter)
    model.finalizeConnections

    # Initialize simulation
    quatTable = osim.TimeSeriesTableQuaternion(sto_filename)
    orientationsData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quatTable)
    oRefs = osim.OrientationsReference(orientationsData)
    init_state = model.initSystem()
    mRefs = osim.MarkersReference()
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    if visualize:
        model.setUseVisualizer(True) # can this be moved below?
    model.initSystem()
    s0 = init_state
    ikSolver = osim.InverseKinematicsSolver(model, mRefs, oRefs, coordinateReferences, constraint_var)#, 1000.)
    ikSolver.setAccuracy = accuracy
    s0.setTime(0.)
    ikSolver.assemble(s0)
    if visualize: # initialize visualization
        model.getVisualizer().show(s0)
        model.getVisualizer().getSimbodyVisualizer().setShowSimTime(True)

    # IK solver loop
    t = 0 # number of steps
    st = 0. # timing simulation
    temp_data = []
    add_time = 0.
    running = True
    start_sim_time = time.time()
    if parallelize: # start another thread to call track
        print("CODE FOR STARTING PARALLEL THREAD")
        ik = Queue() # queue for IMU messages
        ik_list = []
        p_cnt = 0

    q.put(['received']) # tell IMUs to start passing real-time data
    print("Starting recording...")
    while(running):
        if (b.qsize() > 0) or t == sim_steps: # new button press so we should save the data and restart the sim
            if t == sim_steps: # tell IMUs to reset too
                b.put(["done"])
            if log_data:
                ikProc.terminate()
                ik_results = ikReporter.getTable()
                osim.STOFileAdapter.write(ik_results, save_dir+save_file+str(file_cnt)+'.sto')
                np.save(save_dir+ts_file+str(file_cnt)+'.npy', time_vec[:t,:])
                if log_temp:
                    np.save(save_dir+'/tempdata_'+str(file_cnt)+'.npy', temp_data)
                file_cnt += 1
            print("Time used in IK:",st,"Total time:",time.time()-start_sim_time)
            time.sleep(0.5)
            break # exit loop and wait until button pressed for reset
        time_stamp, Qi = q.get()
        add_time = time.time()
        time_s = t*dt
        quat2sto_single(Qi, header_text, sto_filename, time_s, rate, sensor_ind_list) # store next line of fake online data to one-line STO
        
        # IK
        quatTable = osim.TimeSeriesTableQuaternion(sto_filename)
        orientationsData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quatTable)
        rowVecView = orientationsData.getNearestRow(time_s)
        rowVec = osim.RowVectorRotation(rowVecView)
        ikSolver.addOrientationValuesToTrack(time_s+dt, rowVec)
        s0.setTime(time_s+dt)

        if parallelize:
            ikProc1 = Process(target=workers.parallelIK, args=(ikSolver, s0, ik, time_stamp))
            ikProc1.start()
            ik_list.append(ikProc1)
        else:
            ikSolver.track(s0)

        while(ik.qsize() > 0):
            time_vec[p_cnt,1] = ik.get()[0]
            p_cnt += 1
        while len(ik_list) > 5:
            ik_proc = ik_list.pop(0)
            ik_proc.terminate()

        if (p_cnt%int(rate)==1) and log_temp: # log temp data
        #     temp_data.append(cpu.temperature)
            print(cpu.temperature)
            print("Delay (ms):", 1000.*np.mean(time_vec[p_cnt-int(rate):p_cnt,1],axis=0))
        t += 1