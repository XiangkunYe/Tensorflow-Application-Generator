from threading import Thread
from queue import Queue


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ThreadPoolManager(object, metaclass=Singleton):
    def __init__(self, thread_num):
        self.task_queue = Queue()
        self.thread_num = thread_num
        self.__init_threading_pool(self.thread_num)

    def __init_threading_pool(self, thread_num=1):
        for i in range(thread_num):
            thread = ThreadManger(self.task_queue)
            thread.start()

    def add_job(self, func, *args):
        self.task_queue.put((func, args))


class ThreadManger(Thread):
    def __init__(self, task_queue):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.daemon = True

    def run(self):
        while True:
            target, args = self.task_queue.get()
            target(*args)
            self.task_queue.task_done()