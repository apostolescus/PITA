from threading import Lock
import copy
import logging

lock = Lock()
close = False

class RecordStorage:
    permanent = False
    save = False
    smart = True
    fix = False
    record = False
    speed = 0
    start_rec_var = False
    
def initialize(): 
    global num 
    global speed
    global last_val

    speed = 30
    num = 1 
    last_val = 0

def record_mode(mode):
    
    print("rec mode for: ", mode)
    #0 smart mode + auto start
    if mode == 0:
        RecordStorage.record = False
        RecordStorage.smart = True
        RecordStorage.fix = False
        RecordStorage.save = True
        RecordStorage.permanent = False
    #permanent mode
    elif mode == 1:
        RecordStorage.record = False
        RecordStorage.permanent = True
        RecordStorage.fix = False
        RecordStorage.smart = False
        RecordStorage.save = True
    # fix sized
    elif mode == 2:
        RecordStorage.record = False
        RecordStorage.permanent = False
        RecordStorage.fix = True
        RecordStorage.save = True
        RecordStorage.smart = False
    # no record
    else:
        RecordStorage.record = False
        RecordStorage.permanent = False
        RecordStorage.fix = False
        RecordStorage.smart = False
        RecordStorage.save = False
    #print("Record value record_mode: ", RecordStorage.record)
    return True

def start_rec(check = False, start = False, stop = False):

    if stop is True:
        RecordStorage.start_rec_var = False
        return

    if start is True:
        RecordStorage.start_rec_var = True
        return
    
    if check is True:
        return RecordStorage.start_rec_var

def modify_speed(val):
    global speed
    lock.acquire()
    speed = val
    lock.release()

def get_speed():
    global speed
    # lock.acquire()
    # a = copy.deepcopy(speed)
    # lock.release()

    return speed
    
def stop_program():
    global close
    close = True

def should_stop():
    lock.acquire()
    if close is False:
        lock.release()
        return False
    lock.release()
    return True