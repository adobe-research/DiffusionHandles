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
        self.finished_jobs: set[GradioJob] = set()
        self.active_callbacks: list[(Callable[[list[GradioJob], Type], None], list[GradioJob])] = []
        self.finished_callbacks: list[(Callable[[list[GradioJob], Type], None], list[GradioJob])] = []

    def add_job(self, job: GradioJob) -> None:
        self.active_jobs.add(job)
    
    def add_callback(self, func: Callable[[list[GradioJob], Type], None], when_jobs_done: list[GradioJob]):
        if not (set(when_jobs_done) <= (self.active_jobs | self.finished_jobs)):
            raise ValueError("Callback depends on jobs that are not being managed.")
        self.active_callbacks.append((func, when_jobs_done))

    def run(self) -> None:

        while len(self.active_jobs) > 0 or len(self.active_callbacks) > 0:

            newly_finished_jobs = set(job for job in self.active_jobs if job.done())

            self.active_jobs -= newly_finished_jobs
            self.finished_jobs.update(newly_finished_jobs)

            for callback in list(self.active_callbacks): # list will be mutated
                if set(callback[1]).issubset(self.finished_jobs):
                    callback[0](callback[1], self)
                    self.active_callbacks.remove(callback)
                    self.finished_callbacks.append(callback)
            
            if len(self.active_jobs) > 0 or len(self.active_callbacks) > 0:
                time.sleep(0.1)
                for job in self.active_jobs:
                    if job.timed_out():
                        raise TimeoutError("Job timed out.")
