import json
import urllib.request

TFModel_IP = "http://127.0.0.1"
TFModel_PORT = 46176

def train_model_request(request_data):
    url = TFModel_IP + ":" + str(TFModel_PORT)
    post_data = {}
    try:
        post_data['userId'] = request_data['user_id']
        post_data['projectId'] = request_data['project_id']
        post_data['projectDir'] = request_data['path']
        post_data['trainType'] = request_data['ptype']
    except:
        return False, None
    post_data = urllib.parse.urlencode(post_data).encode()
    request = urllib.request.Request(url, data=post_data)
    try:
        response = urllib.request.urlopen(request)
    except Exception as e:
        print("failed: %s" % e)
        return False, None
    charset = response.info().get_content_charset('utf-8')
    content = response.read().decode(charset)
    json_object = json.loads(content)
    return True, json_object

def query_task_state(task_id):
    query_dict = {'taskId': task_id}
    params = urllib.parse.urlencode(query_dict)
    url = TFModel_IP + ":" + str(TFModel_PORT) + "?" + params
    try:
        response = urllib.request.urlopen(url)
    except Exception as e:
        print("failed: %s" % e)
        return False, None
    charset = response.info().get_content_charset('utf-8')
    content = response.read().decode(charset)
    json_object = json.loads(content)
    if 'state' in json_object.keys():
        return True, json_object
    return False, None