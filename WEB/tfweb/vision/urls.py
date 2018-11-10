from django.urls import path
from vision import views
from django.conf.urls import include

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload_file, name='upload'),
    # path('projects/', views.projectListView.as_view(), name='projects'),
    # path('tasks/', views.taskListView.as_view(),name = 'tasks'),
    path('Tutorial/', views.TutorialView, name='tutorial'),
    path('About/', views.AboutView, name='about'),
    path('Contact/', views.ContactView, name='contact'),
    path('Main/', views.MainView, name='main'),
    path('signup/', views.signup, name='signup'),

]