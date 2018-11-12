import json

import rarfile as rarfile
from django.db import connection
from django.http import StreamingHttpResponse
from django.http import HttpResponse
from .models import User, Project, Task, Model
from django.contrib.auth.decorators import login_required
from .train_model import train_model_request, query_task_state, download_model_iterator

def index(request):
    return render(request,'index.html')

def TutorialView(request):
    return render(request,'tutorial.html')

def AboutView(request):
    return render(request, 'about.html')

def ContactView(request):
    return render(request, 'contact.html')

@login_required(login_url='/accounts/login/')
def MainView(request):
    uid = request.user.id
    with connection.cursor() as cursor:
        cursor.execute("SELECT vision_project.id, vision_project.name, vision_task.state , vision_task.id "
                       "FROM vision_project "
                       "LEFT JOIN vision_task on vision_project.id = vision_task.project_id "
                       "where vision_project.user_id = %s", [uid])
        rows = cursor.fetchall()
    projects = []
    for row in rows:
        if row[2] is None:
            projects.append({'id': row[0], 'name': row[1], 'state': 'waiting response', 'task_id': row[3]})
        else:
            projects.append({'id': row[0], 'name': row[1], 'state': row[2], 'task_id': row[3]})
    return render(request, 'mainpage.html', {'username': request.user.username,
                                             'projects': projects})


from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)

            return redirect('/accounts/login/')

    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

@login_required(login_url='/accounts/login/')
def get_task_info(request):
    if request.method == 'GET':
        uid = request.user.id
        with connection.cursor() as cursor:
            cursor.execute("SELECT vision_project.id, vision_project.name, vision_task.state , vision_task.id "
                           "FROM vision_project "
                           "LEFT JOIN vision_task on vision_project.id = vision_task.project_id "
                           "where vision_project.user_id = %s", [uid])
            rows = cursor.fetchall()
        projects = []
        for row in rows:
            if row[2] is None:
                projects.append({'id': row[0], 'name': row[1], 'state': 'waiting response', 'task_id': row[3]})
            else:
                projects.append({'id': row[0], 'name': row[1], 'state': row[2], 'task_id': row[3]})
        return HttpResponse(json.dumps(projects), content_type="application/json")


def update_task_info(request):
    if request.method == 'POST':
        #id = request.POST
        try:
            id = request.POST.get('taskId')
            path = request.POST.get('modelPath', '')
            progress = request.POST.get('progress')
            state = request.POST.get('state')
            task = Task.objects.get(id=id)
            task.path = path
            task.progress = progress
            task.state = state
            task.save()
        except:
            resp = {'errcode': 100, 'detail': 'fail'}
            return HttpResponse(json.dumps(resp), content_type="application/json")
        resp = {'errcode': 200, 'detail': 'success'}
        return HttpResponse(json.dumps(resp), content_type="application/json")

@login_required(login_url='/accounts/login/')
def addProject(request):
    if request.method == 'POST':
        project_name = request.POST["project_name"]
        project_type = request.POST["project_type"]
        from .models import Project
        project = Project()
        project.name = project_name
        project.ptype = project_type
        project.user = request.user
        project.save()
        project_id = project.id
        root_dir = os.getcwd()
        project_path = os.path.join(root_dir, 'media', str(request.user.id), str(project_id))
        os.mkdir(project_path)
        project.path = project_path
        project.save()

        return redirect('/vision/upload?project_id='+ str(project_id))
    return render(request, 'createproject.html')


from django.core.files.storage import FileSystemStorage
import os
import zipfile
@login_required(login_url='/accounts/login/')
def download_file(request):
    task_id = request.GET.get('task_id')
    task = Task.objects.get(id=task_id)
    # TO DO: check auth
    # if task.owner_id != request.user.id:
    #
    file_path = task.path
    file_name = os.path.basename(file_path)
    response = StreamingHttpResponse(download_model_iterator(file_path))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(file_name)

    return response


@login_required(login_url='/accounts/login/')
def upload_file(request):

    if request.method == 'POST' and request.FILES['myfile']:
        project_id = request.GET.get('project_id')
        project = Project.objects.get(id=project_id)
        folder = project.path
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location=folder)
        filename = fs.save(myfile.name, myfile)
        current_path = os.path.join(folder,myfile.name)

        # tar the uploaded file
        if current_path.endswith('zip'):
            zip_file = zipfile.ZipFile(current_path)
            for names in zip_file.namelist():
                zip_file.extract(names, folder)
            zip_file.close()
            extracted_dir = os.path.basename(current_path).split('.')[0]
            project.path = os.path.join(project.path, extracted_dir)
            project.save()

        # request training to TFModel
        project = Project.objects.get(id=project_id)
        request_data = {'user_id': request.user.id,
                        'project_id': project_id,
                        'path': project.path,
                        'ptype': project.ptype}
        isSuccess, task_object = train_model_request(request_data)

        # save & record task
        new_task = Task(id=task_object['taskId'],
                        path='',
                        progress=task_object['progress'],
                        state=task_object['state'],
                        owner=request.user,
                        project=project)
        new_task.save()

        return redirect('/vision/Main')
    else:
        return render(request, 'upload.html')
