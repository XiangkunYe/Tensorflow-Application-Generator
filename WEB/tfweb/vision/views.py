import datetime

from django.shortcuts import render, render_to_response
import os
# Create your views here.
from django.http import HttpResponse

from vision.form import UploadFileForm
from .models import User, Project, Task, Model


def index(request):
    """
    View function for home page of site.
    """
    # Generate counts of some of the main objects
    num_user = User.objects.all().count()
    num_project = Project.objects.all().count()

    num_model = Model.objects.all().count()
    num_task = Task.objects.count()  # The 'all()' is implied by default.

    # Number of visits to this view, as counted in the session variable.
    num_visits = request.session.get('num_visits', 0)
    request.session['num_visits'] = num_visits + 1

    # Render the HTML template index.html with the data in the context variable
    return render(
        request,
        'index.html',
        context={'num_user': num_user, 'num_project': num_project,
                 'num_model': num_model, 'num_task': num_task, 'num_visits':num_visits},
    )

from django.views import generic

class projectListView(generic.ListView):
    model = Project
    template_name = 'project_list.html'


class projectDetailView(generic.DetailView):
    model = Project
    template_name = 'project_detail.html'

class taskListView(generic.ListView):
    model = Task
    template_name = 'task_list.html'


# update
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():#表单数据如果合法
            handle_uploaded_file(request.FILES['file'])#处理上传来的文件
            #return HttpResponse('Successful')
            return render(request, 'returnhome.html')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form':form})

# save
def handle_uploaded_file(f):
    today = str(datetime.date.today())#获得今天日期
    file_name = today + '_' + f.name#获得上传来的文件名称,加入下划线分开日期和名称
    file_path = os.path.join(os.path.dirname(__file__),file_name)#拼装目录名称+文件名称
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)



