import uuid
import json
from thread import TaskManager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 46176
THREAD_NUM = 1


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
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        result = self._parse_post_date(post_data.decode())
        TaskManager().add_job(result)
        self._set_headers()
        self.wfile.write(json.dumps(result).encode())


def server_run(server_class=HTTPServer, handler_class=SelfHTTPHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("serving at port:", port)
    httpd.serve_forever()


if __name__ == '__main__':
    # initialize threadPool
    train_tasks = TaskManager(THREAD_NUM)
    server_run(port=PORT)
