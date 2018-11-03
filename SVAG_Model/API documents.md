# TFModel API document
TFModel is the deep learning model training module in the project Smat Vision App Generator  
## How to deploy TFModel to the server?
to use the TFModel, you should ensure that the python environment has all the packages needed for this module.
you can use requirements.txt to install all the packages:
```bash
pip install -r requirements.txt
```
to start the server:
```bash
python server.py start
```
![Alt text](images_for_README/server_start.PNG?raw=true "Title")  
the TFModel is a daemon peocess, once started it will continue listening to a specific port until you stop it.  
now it will listen to port 46176, you can check with commands below:  
![Alt text](images_for_README/server_working.PNG?raw=true "Title")  
to stop the server:
```bash
python server.py stop
```
also, you can directly kill the process if the stop command doesn't work  
![Alt text](images_for_README/server_stop.PNG?raw=true "Title")  
to restart the server:
```bash
python server.py restart
```
the TFModel will record all the running information in the 'log' file, you can check it if something is wrong.
## How to communicate with TFModel?
Once TFModel is running on the server, it starts to listen a specific Port(which is port 2333 currently)  
It will receive both HTTP GET and POST message from the back-end of our websites  
### to start a training task:
you can simply send HTTP POST message to <server's ip>:2333  
Here are the requirements:  
1. the headers of the HTTP message should include key 'Content-Type' and the value should be 'application/x-www-form-urlencoded'  
2. POST data should include:
```  
    userId: [string]  
    projectId: [string]  
    projectDir: [string]  
    trainType: [string]  
```   
TFModel's response:  
```
taskId: [string]  
progress: [float] 
thread: [string]  
state: [string]   
userId: [string]  
projectId: [string]  
projectDir: [string]  
trainType: [string] 
```
### to query the current state of a task:
you can simply send HTTP GET message to <server's ip>:2333?taskId=<the task's id>  
TFModel's response:
```
taskId: [string]  
progress: [float] 
thread: [string]  
state: [string]   
userId: [string]  
projectId: [string]  
projectDir: [string]  
trainType: [string] 
```