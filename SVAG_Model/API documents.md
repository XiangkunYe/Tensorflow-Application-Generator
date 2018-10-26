# TFModel API document
TFModel is the deep learning model training module in the project Smat Vision App Generator  
## How to deploy TFModel to the server?
...
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