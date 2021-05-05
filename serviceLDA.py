# exec(open("dashboard.py").read())

import subprocess
import time
run = True

p1 = subprocess.Popen(['python3.7',"-m","streamlit","run","./dashboard.py"])
while run == True:
    p0 = subprocess.Popen(['git',"--work-tree=../MicrosoftEssexScrapers","pull"])
    time.sleep(120)
    p2 = subprocess.Popen(['python3.7',"./lda/lda.py"])
    time.sleep(86400)
    p2.kill()
