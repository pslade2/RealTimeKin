#!/usr/bin/python3
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

def clear(q):
    try:
        while True:
            q.get_nowait()
    except:
        pass

# Customize real-time kinematics for use by setting flag and looking at corresponding code below.
real_time = True # set to True for using the kinematics in the python script for real-time applications

# Parameters for IK solver
fake_real_time = False # True to run offline, False to record data and run online
log_temp = True # True to log CPU temperature data
log_data = True # if true save all IK outputs, else only use those in reporter_list for easier custom coding
home_dir = '/home/ubuntu/RealTimeKin/' # location of the main RealTimeKin folder
uncal_model = 'Rajagopal_2015.osim'
uncal_model_filename = home_dir + uncal_model
model_filename = home_dir+'calibrated_' + uncal_model
fake_online_data = home_dir+'recordings/'#test_data.npy'#'test_IMU_data.npy'#'MT_012005D6_009-001_orientations.sto'
sto_filename = home_dir+'tiny_file.sto'
visualize = False
rate = 20.0 # samples hz of IMUs
accuracy = 0.001 # value tuned for accurate and fast IK solver
constraint_var = 10.0 # value tuned for accurate and fast IK solver
init_time = 4.0 # seconds of data to initialize from

# Initialize the quaternions
signals_per_sensor = 6
file_cnt = 0
save_dir_init = home_dir+ 'recordings/' # appending folder name here
save_file = '/recording_'
ts_file = '/timestamp_'
script_live = True

q = Queue() # queue for IMU messages
b = Queue() # queue for button messages
imuProc = Process(target=workers.readIMU, args=(q, b, fake_online_data, init_time, signals_per_sensor, save_dir_init,home_dir))
imuProc.start() # spawning IMU process
sensor_ind_list, rate, header_text, save_folder, save_folder, file_cnt, sim_len, fake_real_time, fake_data_len = b.get()
save_dir = save_dir_init+save_folder+'/' # append the folder name here
kin_store_size = sim_len + 10.0
sim_steps = int(sim_len*rate)
dt = 1/rate

if log_temp and not fake_real_time:
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()

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
    rt_samples = int(kin_store_size*rate)
    #kin_mat = np.zeros((rt_samples, 39)) # 39 is the number of joints stored in the .sto files accessible at each time step
    time_vec = np.zeros((rt_samples,2))
    coordinates = model.getCoordinateSet()
    ikReporter = osim.TableReporter()
    ikReporter.setName('ik_reporter')
    for coord in coordinates:
        if log_data:
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
        model.setUseVisualizer(True)
    model.initSystem()
    s0 = init_state
    ikSolver = osim.InverseKinematicsSolver(model, mRefs, oRefs, coordinateReferences, constraint_var)
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
    q.put(['received']) # tell IMUs to start passing real-time data
    print("Starting recording...")
    while(running):
        if (b.qsize() > 0) or t == sim_steps: # new button press so we should save the data and restart the sim
            if t == sim_steps: # tell IMUs to reset too
                b.put(["done"])
            if log_data:
                ik_results = ikReporter.getTable()
                osim.STOFileAdapter.write(ik_results, save_dir+save_file+str(file_cnt)+'.sto')
                np.save(save_dir+ts_file+str(file_cnt)+'.npy', time_vec[:t,:])
                if log_temp and not fake_real_time:
                    np.save(save_dir+'/tempdata_'+str(file_cnt)+'.npy', temp_data)
                file_cnt += 1
            print("Time used in IK:",st,"Total time:",time.time()-start_sim_time)
            time.sleep(0.5)
            if fake_real_time:
                print("Saved the offline files...")
                exit()
            else:
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
        ikSolver.track(s0)
        if visualize:
            model.getVisualizer().show(s0)
        model.realizeReport(s0)
        if real_time: # The previous kinematics are pulled here and can be used to implement any custom real-time applications
            rowind = ikReporter.getTable().getRowIndexBeforeTime((t+1)*dt) # most recent index in kinematics table
            kin_step = ikReporter.getTable().getRowAtIndex(rowind).to_numpy() # joint angles for current time step as numpy array
            # see the header of the saved .sto files for the names of the corresponding joints.
            ### ADD CUSTOM CODE HERE FOR REAL-TIME APPLICATIONS ###

        st += time.time() - add_time
        time_vec[t,0] = time_stamp
        time_vec[t,1] = time.time()-time_stamp # delay
        if (t%int(rate)==0):
            if fake_real_time: # log temp data
                print(np.round(t*100.0/fake_data_len,1),'%')
            elif log_temp:
                temp_data.append(cpu.temperature)
            #print("Delay (ms):", 1000.*np.mean(time_vec[t-int(rate):t,1],axis=0))
        t += 1
