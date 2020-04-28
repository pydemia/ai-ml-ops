# Configuration file for nbgrader.
# Supervisord is responsible for logging.
c = get_config()

#c.NbGrader.course_id = 'lecture_analytics'
c.Exchange.course_id = 'lecture_analytics'

c.FormgradeApp.authenticator_class = 'nbgrader.auth.hubauth.HubAuth'
c.FormgradeApp.ip = '0.0.0.0'
c.FormgradeApp.port = 5005
c.HubAuth.grader_group = 'formgrade-lecture_analytics'

# c.JupyterHub.services = [
#     {
#         'name': 'formgrade-course1',
#         'admin': True,
#         'url': 'http://0.0.0.0:5005',
#         'user': 'instructor1',
#         'cwd': '/home/pydemia/workspaces',
#         'command': ['nbgrader', 'formgrade']
#     }
# ]
