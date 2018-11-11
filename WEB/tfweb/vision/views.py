import json
from django.db import connection

from django.http import HttpResponse
from .models import User, Project, Task, Model
from django.contrib.auth.decorators import login_required


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
        cursor.execute("SELECT svag_db.vision_project.id, svag_db.vision_project.name, svag_db.vision_task.state "
                       "FROM svag_db.vision_project "
                       "left join svag_db.vision_task on svag_db.vision_project.id = svag_db.vision_task.project_id "
                       "where svag_db.vision_project.user_id = %s", [uid])
        rows = cursor.fetchall()
    projects = []
    for row in rows:
        if row[2] is None:
            projects.append({'id': row[0], 'name': row[1], 'state': 'waiting response'})
        else:
            projects.append({'id': row[0], 'name': row[1], 'state': row[2]})
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


def get_task_info(request):
    if request.method == 'GET':
        project_id = request.GET.get('projectId', default='')
        if project_id == '':
            resp = {'errcode': 100, 'detail': 'invalid project_id'}
            return HttpResponse(json.dumps(resp), content_type="application/json")
        try:
            task = Task.objects.get(project_id=project_id)
            task_info = {'id': str(task.id),
                         'path': task.path,
                         'progress': task.progress,
                         'owner_id': str(task.owner_id),
                         'project_id': str(task.project_id),
                         'state': task.state}
            resp = {'errcode': 200, 'detail': '', 'taskInfo': task_info}
            return HttpResponse(json.dumps(resp), content_type="application/json")
        except:
            resp = {'errcode': 101, 'detail': 'get task info failed'}
            return HttpResponse(json.dumps(resp), content_type="application/json")

def update_task_info(request):
    if request.method == 'POST':
        #id = request.POST
        try:
            id = request.POST.get('taskId')
            path = request.POST.get('modelPath')
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

        return redirect('/vision/upload?project_id='+ str(project_id))
    return render(request, 'createproject.html')


from django.core.files.storage import FileSystemStorage
import os

def upload_file(request):

    if request.method == 'POST' and request.FILES['myfile']:
        project_id = request.GET.get('project_id')
        folder = os.path.join('media/', str(request.user.id),str(project_id))
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location=folder)
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'upload.html')