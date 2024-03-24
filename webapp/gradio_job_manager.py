import time
from typing import Callable, Type

import gradio_client
import gradio_client.client


class GradioJob():
    """
    Wrapper around a gradio job that also keeps track of time.
    """
    def __init__(
            self, job: gradio_client.client.Job,
            # finalize_func: Callable[[list, Type], None] = None,
            timeout: float = None):

        self.job = job
        self.start_time = time.time()
        self.timeout = timeout

    def done(self) -> bool:
        return self.job.done()

    def outputs(self) -> list:
        return self.job.outputs()

    def timed_out(self) -> bool:
        return self.timeout is not None and time.time() - self.start_time > self.timeout

class GradioJobManager():
    def __init__(self):
        self.active_jobs: set[GradioJob] = set()
        self.callbacks: list[(Callable[[list[GradioJob], Type], None], list[GradioJob])] = []

    def add_job(self, job: GradioJob) -> None:
        self.active_jobs.add(job)
    
    def add_callback(self, func: Callable[[list[GradioJob], Type], None], when_jobs_done: list[GradioJob]):
        self.callbacks.append((func, when_jobs_done))
    
    def run(self) -> None:

        while len(self.active_jobs) > 0:

            finished_jobs = set(job for job in self.active_jobs if job.done())

            callback_jobs = set()
            for callback in list(self.callbacks): # list will be mutated
                if set(callback[1]).issubset(finished_jobs):
                    callback[0](callback[1], self)
                    self.callbacks.remove(callback)
                else:
                    callback_jobs.update(callback[1])
            
            # remove jobs that have fininshed and are not used in a callback
            self.active_jobs -= (finished_jobs - callback_jobs)

            if len(self.active_jobs) > 0:
                time.sleep(0.1)
                for job in self.active_jobs:
                    if job.timed_out():
                        raise TimeoutError("Job timed out.")
