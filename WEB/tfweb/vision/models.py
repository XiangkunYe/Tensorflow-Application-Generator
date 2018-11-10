from django.db import models
from django.conf import settings
import uuid # Required for unique user_id
# Create your models here.
class User(models.Model):
    """
    Model representing a user
    """
    name = models.CharField(max_length=100, help_text="Enter a user name")
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, help_text="Unique ID for a user")
    email = models.EmailField(max_length=254,help_text="Enter a email address")
    password = models.CharField(max_length=20,help_text="Enter the password")

    def __str__(self):
        """
        String for representing the Model object (in Admin site etc.)
        """

        return self.name
class Project(models.Model):
    """
    Model representing a project
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, help_text="Unique ID for a project")
    name = models.CharField(max_length=1000)
    ptype = models.CharField(max_length=50)
    path = models.CharField(max_length=1000)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        """
        String for representing the project.
        """
        return self.name

class Model(models.Model):
    """
    Model representing a Ml model
    """
    mtype = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    def __str__(self):
        """
        String for representing the model name .
        """
        return self.name

class Task(models.Model):
    """
    Model representing a Ml task
    """
    id = models.CharField(primary_key=True,max_length=254)
    path = models.CharField(max_length=1000)
    progress = models.CharField(max_length=20)
    state = models.CharField(max_length=32)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    project = models.ForeignKey('Project', on_delete=models.SET_NULL, null=True)

    def __str__(self):
        """
        String for representing the task .
        """
        return self.id


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)



