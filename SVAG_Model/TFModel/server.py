import uuid
import json
import platform
import time
import os
import logging
import sys
from thread import TaskManager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# daemonize process is not supported on Windows
if platform.system() != "Windows":
    import daemon
    import lockfile

PORT = 46176
THREAD_NUM = 1
LOCAL_PATH = os.getcwd()


class SelfHTTPHandler(BaseHTTPRequestHandler):
    def _parse_post_date(self, post_data):
        result = {}
        post_data = parse_qs(post_data)
        # the values of post_data[key_name] is a list of values with key_name
        result['userId'] = post_data.get('userId', [''])[0]
        result['projectId'] = post_data.get('projectId', [''])[0]
        result['projectDir'] = post_data.get('projectDir', [''])[0]
        result['trainType'] = post_data.get('trainType', [''])[0]
        result['taskId'] = str(uuid.uuid1())
        result['state'] = 'initialize'
        result['progress'] = 0
        result['thread'] = ''
        return result

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        server_logger = logging.getLogger('main_server')
        d = {'clientip': self.client_address[0]}
        server_logger.info("recv GET request", extra=d)
        parsed_path = urlparse(self.path)
        parsed_path = parse_qs(parsed_path.query)
        task_id = parsed_path.get('taskId', ['no taskId in query'])[0]
        info = TaskManager().get_task_info(task_id)
        if len(info) == 0:
            info['error'] = 'cannot find task info'
        info['taskId'] = task_id
        self._set_headers()
        self.wfile.write(json.dumps(info).encode())

    def do_POST(self):
        server_logger = logging.getLogger('main_server')
        d = {'clientip': self.client_address[0]}
        server_logger.info("recv POST request", extra=d)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        result = self._parse_post_date(post_data.decode())
        server_logger.info("initialize task id, added to tasks pool", extra=d)
        TaskManager().add_job(result)
        self._set_headers()
        self.wfile.write(json.dumps(result).encode())


def initialize_server_logger():
    with open(os.path.join(LOCAL_PATH, 'log', 'server.log'), 'w+') as log_file:
        log_file.write("*******************start server at {}*******************\n".format(
            time.asctime(time.localtime(time.time()))))

    server_logger = logging.getLogger('main_server')

    server_logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(LOCAL_PATH, 'log', 'server.log'))
    fh.setLevel(logging.DEBUG)

    server_formatter = logging.Formatter('%(asctime)s - %(clientip)s - %(message)s')
    fh.setFormatter(server_formatter)
    server_logger.addHandler(fh)


def server_run(server_class=HTTPServer, handler_class=SelfHTTPHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("serving at port:", port)
    httpd.serve_forever()


def main():
    # intialize server logger
    initialize_server_logger()
    # initialize task threading pool
    _ = TaskManager(THREAD_NUM)
    server_run(port=PORT)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(LOCAL_PATH, 'log')):
        os.mkdir(os.path.join(LOCAL_PATH, 'log'))

    # daemonize process is not supported on Windows
    if platform.system() == "Windows":
        main()
    else:
        with open(os.path.join(LOCAL_PATH, 'log', 'server_output.log'), 'w+') as log_file:
            with daemon.DaemonContext(pidfile=lockfile.FileLock(os.path.join(LOCAL_PATH, 'TFModel_server.pid')),
                                      stdout=log_file,
                                      stderr=log_file):
                main()
