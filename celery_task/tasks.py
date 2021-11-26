import importlib
from abc import ABC

from celery import Task

from .worker import app


class PredictTask(Task, ABC):
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call
        """

        if not self.model:
            module_import = importlib.import_module(self.path[0])
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj()

        return self.run(*args, **kwargs)


@app.task(ignore_result=False,
          base=PredictTask,
          bind=True,
          name="predict_task",
          path=("celery_task.executor.model", "Executor"))
def submit_task(self, data):
    """
    Submit task to celery
    """
    return self.model.process(data)
