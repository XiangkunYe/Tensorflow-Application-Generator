import threading
from threading import Thread
from queue import Queue


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TaskManager(object, metaclass=Singleton):
    def __init__(self, thread_num=1):
        self.task_queue = Queue()
        self.thread_num = thread_num
        self.__init_threading_pool(self.thread_num)
        self.task_dict = {}

    def get_task_info(self, task_id):
        return self.task_dict.get(task_id, {})

    def __init_threading_pool(self, thread_num):
        for i in range(thread_num):
            thread = TaskThread(self.task_queue)
            thread.start()

    def add_job(self, task_info):
        self.task_dict[task_info['taskId']] = {
            'thread': '',
            'progress': 0,
            'userId': task_info['userId'],
            'projectId': task_info['projectId'],
            'userDir': task_info['userDir'],
            'trainType': task_info['trainType'],
            'state': 'pending',
            'modelPath': ''
        }
        print("recv and add a new task: {}".format(task_info['taskId']))
        self.task_queue.put(task_info['taskId'])


from train import train


class TaskThread(Thread):
    def __init__(self, task_queue):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.daemon = True
        self.task_id = ''

    def run(self):
        while True:
            self.task_id = self.task_queue.get()
            TaskManager().task_dict[self.task_id]['state'] = 'processing'
            method = TaskManager().task_dict[self.task_id]['trainType']
            user_dir = TaskManager().task_dict[self.task_id]['userDir']
            print("start training (thread:%s)" % threading.current_thread().name)
            model_file_path = train(self.task_id, method, user_dir)
            print("finish (thread:%s)" % threading.current_thread().name)
            TaskManager().task_dict[self.task_id]['state'] = 'done'
            TaskManager().task_dict[self.task_id]['modelPath'] = model_file_path
            self.task_queue.task_done()
