from .models import User, Project, Task, Model
from django.contrib.auth.decorators import login_required


def index(request):
    """
    View function for home page of site.
    """
    # num_user = User.objects.all().count()
    # num_project = Project.objects.all().count()
    # num_model = Model.objects.all().count()
    # num_task = Task.objects.count()
    # num_visits = request.session.get('num_visits', 0)
    # request.session['num_visits'] = num_visits + 1

    # Render the HTML template index.html with the data in the context variable
    return render(request,'index.html',)


from django.core.files.storage import FileSystemStorage
import os

def upload_file(request):
    folder = os.path.join('media/',str(request.user.id),str(Project.id))
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location=folder)
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'upload.html')


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
            return redirect('/accounts/login/')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})