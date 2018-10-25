import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

PORT = 2333


class SelfHTTPHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        post_data = parse_qs(post_data)
        print(post_data)
        self._set_headers()

def server_run(server_class=HTTPServer, handler_class=SelfHTTPHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("serving at port:", port)
    httpd.serve_forever()

if __name__ == '__main__':
    server_run(port=PORT)
