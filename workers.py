import helper as h
import time
import sys
import ahrs
import numpy as np
import os

def parallelIK(ikSolver, s0, ik, time_stamp):
    ikSolver.track(s0)
    ik.put([time.time()-time_stamp])
    time.sleep(0.005)

def readIMU(q, b, fake_online_data, init_time, signals_per_sensor, save_dir_init,home_dir):
    # Load the initialization information about the sensors
    tca_inds = []
    num_parts = 0
    calibrate_sensors = False
    parallelize = False
    old_lines = []
    save_folder = 'test_dir'
    sim_len = 600
    # Defining the external signal trigger
    imu_only = False
    with open(home_dir+'settings.txt', 'r') as f:
        for cnt, line in enumerate(f):
            old_lines.append(line)
            if cnt == 0:
                body_parts = line.split(',')
                num_parts = len(body_parts)
            elif cnt == 1:
                tca_inds = line.split(',')
                if num_parts != len(tca_inds):
                    print("Wrong number of tca_indeces given, doesn't match number of body parts.")
                alt_address_list = []
                tca_inds = tca_inds[:-1]
                for i in range(len(tca_inds)):
                    if len(tca_inds[i]) == 1: # alternate
                        tca_inds[i] = int(tca_inds[i])
                        alt_address_list.append(False)
                    elif len(tca_inds[i]) > 1:
                        tca_inds[i] = int(tca_inds[i][0])
                        alt_address_list.append(True)
            elif cnt == 2:
                rate = float(line)
                print("Rate:",rate)
            elif cnt == 6:
                cal_word = line.strip()
                if cal_word == 'calibrate': # calibrate IMUs at start
                    calibrate_sensors = True
            elif cnt == 3:
                cal_word = line.strip()
                if cal_word == 'parallel': # run with extra thread multiprocessing
                    parallelize = True
                    fake_real_time = False
                elif cal_word == 'online': # run offline with given file path in recordings folder
                    fake_real_time = False
                elif cal_word == 'offline':
                    fake_real_time = False
                    imu_only = True
                else:
                    fake_path = cal_word
                    fake_real_time = True
            elif cnt == 4:
                cal_word = line.strip()
                save_folder = cal_word
            elif cnt == 5:
                sim_len = float(line)
                print("Sim length:",sim_len)
    f.close()
    if calibrate_sensors:
        with open(home_dir+'settings.txt', 'w') as f:
            f.writelines(old_lines[:-1])
        f.close()

    if not fake_real_time:
        from adafruit_lsm6ds import ISM330DHCT, Rate, AccelRange, GyroRange
        import adafruit_tca9548a
        import board
        import busio
        import digitalio
        from micropython import const
        from adafruit_bus_device.i2c_device import I2CDevice
        trigger = digitalio.DigitalInOut(board.D16) # external signal should be applied to the BCM 16 pin
        trigger.direction = digitalio.Direction.INPUT # this signal will be checked, if 3.3V is applied, recording will be started
        trigger.pull = digitalio.Pull.DOWN # pull this input low at all times
        trigger_status = False # set to true if the trigger is used to start a recording
        # Initializing the different methods
        button_address = const(0x6F) # I2c address for LED button
        i2c = busio.I2C(board.SCL, board.SDA)
        tca = adafruit_tca9548a.TCA9548A(i2c)
        button = I2CDevice(tca[tca_inds[0]], button_address)
        last_pressed = time.time() - 1.0
        pressed = False
        button_mode(button, 0) # turn button off
        clear_button(button)
    # define sensors
    sensor_inds = tca_inds[1:]
    alt_address_list = alt_address_list[1:]
    sensor_list = []
    sensor_ind_list = []
    sensor_number = []
    sensor_cnt = 0
    sensor_rot = []
    sensor_rot_type = [0,0,1,1,3,2,2,3,1,1,1,2,2,2] # define rotation types
    sensor_labels_full = ['pelvis_imu','torso_imu','femur_l_imu','tibia_l_imu','calcn_l_imu','femur_r_imu','tibia_r_imu','calcn_r_imu','humerus_l_imu','ulna_l_imu','hand_l_imu','humerus_r_imu','ulna_r_imu','hand_r_imu']
    sensor_label_list = []
    for i, s_ind in enumerate(sensor_inds):
        if s_ind != 9:
            if not fake_real_time:
                if alt_address_list[i]: # if true use alternate address
                    s = ISM330DHCT(tca[s_ind], address=const(0x6B))
                else:
                    s = ISM330DHCT(tca[s_ind])
                sensor_list.append(s)
            sensor_ind_list.append(s_ind)
            len_sensor_list = len(sensor_ind_list)
            sensor_number.append(sensor_cnt)
            sensor_cnt += 1
            sensor_rot.append(sensor_rot_type[i]) # say for this number sensor how to rotate it
            sensor_label_list.append(sensor_labels_full[i])
            
    # Making the text header for which body segments have IMU data
    header_text = 'time\t'
    for label in sensor_label_list:
        header_text = header_text + '\t' + label
    header_text = header_text + '\n'
    num_sensors = len(sensor_number)
    if not fake_real_time:
        for s in sensor_list: # setting all sensors to same default values
            s.accelerometer_range = AccelRange.RANGE_8G
            s.gyro_range = GyroRange.RANGE_2000_DPS
            # To change the imu data sampling frequency, use the lines below. 104 Hz is default.
            #s.accelerometer_data_rate = Rate.RATE_416_HZ # Other options: 52_HZ, 26_HZ, 104_HZ, 416_HZ
            #s.gyroscope_data_rate = Rate.RATE_416_HZ

    # load fake data and figure out number of sensors
    quat_cal_offset = int(init_time*rate) # array for data for calibrating sensors
    cwd = os.getcwd() #
    sensor_vec = np.zeros(num_sensors*signals_per_sensor)
    scaling = np.ones(num_sensors*signals_per_sensor)
    offsets = np.zeros(num_sensors*signals_per_sensor)
    imu_data = np.zeros((quat_cal_offset, num_sensors*signals_per_sensor))
    fake_data_len = 0
    if fake_real_time:
        cal_data = imu_data
        imu_data = np.load(fake_online_data + fake_path) # load fake dataset
        cal_data = imu_data[:quat_cal_offset,:]
        fake_data_len = imu_data.shape[0]
        print("Starting offline analysis for file with",fake_data_len,"samples")
    # calibrating or loading calibration data
    if not fake_real_time:
        cal_dir = home_dir+'calibration'
        gyro_file = '/gyro_offsets.npy'
        if calibrate_sensors or not os.path.exists(cal_dir): # also check if calibration folder exists
            print("Calibrating sensors!")
            try: # create calibration dir
                os.makedirs(cal_dir)
            except:
                pass
            calibrating_sensors(cal_dir, gyro_file, button, rate, sensor_list)
            button_mode(button, 0) # turn button off
        offsets = np.load(cal_dir+gyro_file)# loading calibration vec
    else:
        offsets = 0.0
    save_dir = save_dir_init+save_folder+'/' # append the folder name here
    file_cnt = 0
    try: # create save dir or count number of files so I don't save over some
        os.makedirs(save_dir)
    except:
        f = os.listdir(save_dir)
        for s in f:
            if 'rec' in s:
                file_cnt += 1

    b.put([sensor_number, rate, header_text, parallelize, save_folder, file_cnt, sim_len, fake_real_time,fake_data_len,]) # ready to start running
    if fake_real_time:
        time.sleep(2.)
        for i in range(quat_cal_offset):# pull in real data and compute quats for init_time
            cal_data[i,:] = imu_data[0,:]
        Qi, head_err, rot_mats = h.compute_quat(cal_data, len_sensor_list, quat_cal_offset, sensor_rot, num_sensors)

        q.put([time.time(), Qi, head_err]) # sending initialized info
        time_start = time.time()
        dt = 1/rate
        madgwick = ahrs.filters.Mahony(frequency=rate)
        t = 0
        sensor_vec = np.zeros(num_sensors*signals_per_sensor)
        start = q.get() # waiting for confirmation of sim Starting
        time.sleep(0.3)
        while(t < fake_data_len): # Pull data at the desired rate
            sensor_vec = imu_data[t,:]
            for i in range(len_sensor_list):
                s_off = i*signals_per_sensor
                accel = np.matmul(sensor_vec[s_off:s_off+3],rot_mats[i,:,:])
                gyro = np.matmul(sensor_vec[s_off+3:s_off+6],rot_mats[i,:,:])
                Qi[i,:] = madgwick.updateIMU(Qi[i,:], gyro, accel)
            while(q.qsize()>0):
                time.sleep(0.003)
            q.put([time.time(), Qi])
            t += 1
        b.put([True]) # end the script
    else:
        while(True): # outer loop for resetting the simulation
            button_mode(button, 1) # make button blink
            clear_button(button)
            while(not pressed): # wait for button press
                pressed, last_pressed = check_button(button, last_pressed)
                if trigger.value: # External signal pulled up to 3.3V to start recording
                    pressed = True
                    trigger_status = True
            for i in range(quat_cal_offset):# pull in real data and compute quats for init_time
                for j, s in enumerate(sensor_list):
                    s_off = j*signals_per_sensor
                    imu_data[i, s_off:s_off+3] = s.acceleration
                    imu_data[i, s_off+3:s_off+6] = s.gyro + offsets[s_off+3:s_off+6] 
                    
            imu_data[i,:] = imu_data[i,:] + offsets # correcting gyro bias
            Qi, head_err, rot_mats = h.compute_quat(imu_data, len_sensor_list, quat_cal_offset, sensor_rot, num_sensors)

            q.put([time.time(), Qi, head_err]) # sending initialized info
            time_start = time.time()
            dt = 1/rate
            madgwick = ahrs.filters.Mahony(frequency=rate)
            t = 0
            sensor_vec = np.zeros(num_sensors*signals_per_sensor)
            sensor_mat = np.zeros((int(sim_len*rate),num_sensors*signals_per_sensor))
            start = q.get() # waiting for confirmation of sim Starting
            time.sleep(0.3)
            button_mode(button, 2) # make button solid red to start recording
            clear_button(button)
            while(True): # Pull data at the desired rate
                cur_time = time.time()
                if cur_time >= time_start + dt: # time for next reading
                    pressed, last_pressed = check_button(button, last_pressed)
                    if trigger_status: # trigger was engaged
                        if not trigger.value: # the external signal is now low (0V), stop recording
                            pressed = True
                    if pressed or (b.qsize() > 0): # send message to exit the recording
                        b.put([pressed])
                        q.put([cur_time, Qi])
                        button_mode(button, 0) # turn button off
                        np.save(save_dir+'raw_imu_'+str(file_cnt)+'.npy', sensor_mat[:t,:]) # saving kinematics
                        file_cnt += 1
                        pressed = False
                        time.sleep(1.0)
                        break
                    time_start = cur_time
                    for j, s in enumerate(sensor_list):
                        s_off = j*signals_per_sensor
                        sensor_vec[s_off:s_off+3] = s.acceleration
                        sensor_vec[s_off+3:s_off+6] = s.gyro
                    sensor_vec = sensor_vec + offsets # preping
                    sensor_mat[t,:] = sensor_vec
                    for i in range(len(sensor_list)):
                        s_off = i*signals_per_sensor
                        accel = np.matmul(sensor_vec[s_off:s_off+3],rot_mats[i,:,:])
                        gyro = np.matmul(sensor_vec[s_off+3:s_off+6],rot_mats[i,:,:])
                        Qi[i,:] = madgwick.updateIMU(Qi[i,:], gyro, accel)
                    if not imu_only:
                        q.put([cur_time, Qi])
                    t += 1

def button_mode(button, state, ON=0xFF, OFF = 0x00, LED=0x0F, b_cycle_time=0x1B, b_brightness=0x19, b_off_time=0x1D):
    with button:
        if state == 0: # turn off
            button.write(bytes([b_brightness, OFF]), stop=False)
            button.write(bytes([b_off_time, OFF]), stop=False)
            button.write(bytes([b_cycle_time, OFF]), stop=False)
        elif state == 1: # blink
            button.write(bytes([b_brightness, LED]), stop=False)
            button.write(bytes([b_off_time, ON]), stop=False)
            button.write(bytes([b_cycle_time, ON]), stop=False)
        elif state == 2: # solid red
            button.write(bytes([b_brightness, LED]), stop=False)
            button.write(bytes([b_off_time, OFF]), stop=False)
            button.write(bytes([b_cycle_time, OFF]), stop=False)
        else:
            print("Unknown button state")

# checks for button press, have to wait at least time_min seconds before pressing again
def check_button(button, last_pressed, time_min = 1.0, PRESSED = 0x03, OFF = 0x00):
    state = False
    with button:
        button.write(bytes([PRESSED]), stop=False)
        result = bytearray(1)
        button.readinto(result)
        #print(result)
        if result != bytes([OFF]) and (time.time()-last_pressed > time_min): # pressed
            state = True
            last_pressed = time.time()
            button.write(bytes([PRESSED, OFF]), stop=False) # reset
        else:
            button.write(bytes([PRESSED, OFF]), stop=False) # reset
    return state, last_pressed

def clear_button(button, PRESSED = 0x03, OFF = 0x00):
    with button:
        button.write(bytes([PRESSED, OFF]), stop=False) # reset

def calibrating_sensors(cal_dir, gyro_file, button, rate, sensor_list, calibration_time=10.0, signals_per_sensor=6, b_brightness=0x19):
    dt = 1/rate
    num_samples = int(calibration_time//dt)
    num_sensors = len(sensor_list)
    cal_data = np.zeros((num_samples, 6*num_sensors))
    time_start = time.time()
    led_range = 255
    sample_cnt = 0
    while sample_cnt < num_samples:
        cur_time = time.time()
        if cur_time >= time_start + dt: # time for next reading
            time_start = cur_time
            for j, s in enumerate(sensor_list):
                s_off = j*signals_per_sensor
                cal_data[sample_cnt, s_off+3:s_off+6] = s.gyro
            with button:
                button.write(bytes([b_brightness, sample_cnt*8%led_range]), stop=False)
            sample_cnt += 1
    gyro_offset = -1.0*np.mean(cal_data,axis=0)
    np.save(cal_dir+gyro_file, gyro_offset)
