from django.urls import path
from vision import views
from django.conf.urls import include

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload_file, name='upload'),
    path('Tutorial/', views.TutorialView, name='tutorial'),
    path('About/', views.AboutView, name='about'),
    path('Contact/', views.ContactView, name='contact'),
    path('Main/', views.MainView, name='main'),
    path('signup/', views.signup, name='signup'),
    path('taskInfo/', views.get_task_info, name='taskInfo'),
    path('taskUpdate', views.update_task_info, name='taskUpdate')
]