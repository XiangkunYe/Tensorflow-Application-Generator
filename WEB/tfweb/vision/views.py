import datetime

from django.shortcuts import render, render_to_response
import os
# Create your views here.
from django.http import HttpResponse

from tfweb.settings import LOGIN_REDIRECT_URL
from vision.form import DocumentForm
from .models import User, Project, Task, Model
from django.contrib.auth.decorators import login_required


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
    return render(request,'index.html',)

# class projectListView(generic.ListView):
#     model = Project
#     template_name = 'project_list.html'
#
#
#
# class taskListView(generic.ListView):
#     model = Task
#     template_name = 'task_list.html'


def upload_file(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect(LOGIN_REDIRECT_URL)
    else:
        form = DocumentForm()
    return render(request, 'upload.html', {
        'form': form
    })


def TutorialView(request):
    return render(request,'tutorial.html')

def AboutView(request):
    return render(request, 'about.html')

def ContactView(request):
    return render(request, 'contact.html')

@login_required(login_url='/accounts/login/')
def MainView(request):
    uid = request.user.id
    projects = Project.objects.filter(user_id=uid)
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
            #return redirect('home')
            return redirect('http://127.0.0.1:8000/accounts/login/')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})