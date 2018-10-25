import uuid
import json
from TFModel.thread import ThreadPoolManager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

PORT = 2333
THREAD_NUM = 1


class SelfHTTPHandler(BaseHTTPRequestHandler):
    def _parse_post_date(self, post_data):
        result = {}
        post_data = parse_qs(post_data)
        # the values of post_data[key_name] is a list of values with key_name
        result['userId'] = post_data.get('userId', '')[0]
        result['projectId'] = post_data.get('projectId', '')[0]
        result['dataDir'] = post_data.get('dataDir', '')[0]
        result['trainType'] = post_data.get('trainType', '')[0]
        result['taskId'] = str(uuid.uuid1())
        result['progress'] = 0
        result['state'] = 'initialize'
        return result

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        result = self._parse_post_date(post_data.decode())
        thread_pool = ThreadPoolManager()
        thread_pool.add_job(start_train, result)
        self._set_headers()
        self.wfile.write(json.dumps(result).encode())


def server_run(server_class=HTTPServer, handler_class=SelfHTTPHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("serving at port:", port)
    httpd.serve_forever()

if __name__ == '__main__':
    # initialize threadPool
    thread_pool = ThreadPoolManger(THREAD_NUM)
    server_run(port=PORT)
