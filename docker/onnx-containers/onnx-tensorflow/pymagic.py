import os
import sys
from IPython import get_ipython
from IPython.core.magics import ScriptMagics, PackagingMagics
from IPython.core.magic import Magics, magics_class, line_magic

#%alias python "sys.executable"
#%alias_magic python run

# ------------------------------------------- #
OLD_PATH = os.environ['PATH']
ENV_PATH = os.path.dirname(sys.executable)
NEW_PATH = ":".join([ENV_PATH, OLD_PATH])


def _activate_env(f):
    os.environ['PATH'] = NEW_PATH
    return f


# ------------------------------------------- #
#@magics_class
class EnvScriptMagics(ScriptMagics):

    def __init__(self, *args, **kwargs):
        super(ScriptMagics, self).__init__(*args, **kwargs)

    @_activate_env
    def shebang(self, line, cell):
        super(ScriptMagics, self).shebang(line, cell)


# # ------------------------------------------- #
@magics_class
class EnvRunMagics(PackagingMagics):

    @line_magic("python")
    def python(self, line):
        self.shell.system(' '.join([sys.executable, line]))


def load_ipython_extension(ipython):

    ipython.register_magics(EnvScriptMagics)
    ipython.register_magics(EnvRunMagics)
