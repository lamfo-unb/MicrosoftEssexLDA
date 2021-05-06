# exec(open("dashboard.py").read())

import subprocess
import time
run = True
import psutil, os



# p1 = subprocess.Popen(['python3.7',"-m","streamlit","run","./dashboard.py"])
# while run == True:
#     p0 = subprocess.Popen(['git',"--work-tree=../MicrosoftEssexScrapers","pull"])
#     time.sleep(120)
#     p2 = subprocess.Popen(['python3.7',"./lda/lda.py"])
#     time.sleep(86400)
#     p2.kill()


#https://www.shellhacks.com/windows-taskkill-kill-process-by-pid-name-port-cmd/
#C:\> tasklist | findstr /I process_name

#https://www.geeksforgeeks.org/kill-a-process-by-name-using-python/



#https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform

def lowpriority():
    import os, signal

    """ Set the priority of the process to below-normal."""

    # Ask user for the name of process
    name = "Nice"
    try:
        pids = []  
        # iterating through each instance of the proess
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"): 
            fields = line.split()
              
            # extracting Process ID from the output
            pid = fields[0] 
            pids.append(pid)
              
            # terminating process 
            # os.kill(int(pid), signal.SIGKILL) 
          
    except Exception as E:
        # print(E)
        py = subprocess.Popen(["tasklist","|","findstr","/I","process_name"])
        print(py)
    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        # pid = win32api.GetCurrentProcessId()
        for pid in pids:
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, int(pid))
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os
        for pid in pids:
            px = psutil.Process(int(pid))
            print(px.nice())
            print(px.nice(10))
            print(px.nice())


lowpriority()