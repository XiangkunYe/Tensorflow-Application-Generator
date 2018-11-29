import urllib.parse
import urllib.request

WEB_SERVER_IP = "http://127.0.0.1"
WEB_SERVER_PORT = 2333
WEB_SERVER_ROUTE = '/vision/taskUpdate'

def update_task_info(task_id, task_info):
    url = WEB_SERVER_IP + ":" + str(WEB_SERVER_PORT) + WEB_SERVER_ROUTE
    post_data = task_info
    post_data['taskId'] = task_id
    post_data = urllib.parse.urlencode(post_data).encode()
    request = urllib.request.Request(url, data=post_data)
    try:
        urllib.request.urlopen(request)
    except Exception as e:
        print("failed: %s" % e)
        return False
    return True
