from django.contrib import admin

# Register your models here.
from .models import User, Project, Model,Task

#admin.site.register(User)
# Register the admin class with the associated model
# Define the admin class

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
  list_display = ('name', 'id', 'email', 'password')
  fieldsets = (
      (None, {
          'fields': ('name', 'id')
      }),
      ('Account', {
          'fields': ('email', 'password')
      }),
  )


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'ptype', 'path','user')


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):

    list_filter = ('mtype','name')

admin.site.register(Task)



