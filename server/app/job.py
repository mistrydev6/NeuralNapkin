from dataclasses import asdict
from enum import Enum
import json
import sys
import time
from typing import cast
from pydantic.v1.env_settings import DotenvType
import redis
import logging
import threading
from typing import Any, Tuple
from utils import (
    CONTROLNET_NAME,
    BGREM_NAME,
    CRM_NAME,
    ControlNetPrediction,
    BgRemovalPrediction,
    CRMPrediction,
    Prediction,
    PredictionOutput,
    PredictionStatus,
)

import replicate

replicate.default_client._api_token = ""  # type: ignore

r = redis.Redis(decode_responses=True)
JOB_TOPIC = "job-1"


class JobStatus:
    Pending = "pending"
    InProgress = "in-progress"
    Done = "done"
    Canceled = "canceled"
    Failed = "failed"


def add_job(gid: str, img_b64: str, word: str):
    # FIXME: Assert b64
    r.set(gid, json.dumps({"status": JobStatus.Pending, "data": img_b64, "word": word}))
    r.publish(JOB_TOPIC, gid)


def get_job(gid: str):
    return json.loads(cast(str, r.get(gid)))


class ProcessingStep(Enum):
    ControlNet = 0
    BgRemoval = 1
    CRM = 2


class ProcessingStatus(Enum):
    Pending = 0
    Processing = 1
    Done = 2


class JobExecutor:
    def __init__(self) -> None:
        # Setup logger
        self._logger = logging.getLogger("job-executor")
        sh = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(args)s"
        )
        sh.setFormatter(formatter)
        self._logger.addHandler(sh)
        self._logger.setLevel(logging.DEBUG)

        # Mutex handler
        self._lock = threading.Lock()
        self._inbox = {}

        self._in_progress_jobs = {}
        self._r = redis.Redis(decode_responses=True)

    def _load_jobs(self):
        self._logger.info("Start loading inbox")
        gids = cast(list, self._r.keys())
        for gid in gids:
            job = self._get_job(gid)
            if job.get("status") in [JobStatus.Pending, JobStatus.InProgress]:
                with self._lock:
                    self._inbox[gid] = job
                    # TODO: This is a hack. Better do properly parse json data to Predictions
                    job["logs"] = []
            else:
                # TODO: Expire job
                pass

        self._logger.info(f"Loadded {len(gids)} jobs")

    def _start_listener(self):
        def process():
            while True:
                logger = self._logger.getChild("job-listener")

                logger.info("Starting job listener...")

                pubsub = self._r.pubsub()
                pubsub.subscribe(JOB_TOPIC)

                for msg in pubsub.listen():
                    try:
                        logger.debug("Processing new message", msg)
                        # Listen to only message
                        if msg.get("type") != "message":
                            continue

                        gid = msg.get("data")
                        if gid is not None:
                            job = self._get_job(gid)

                            # TODO: Find a better way to validate job
                            if len(job) == 0:
                                continue
                            with self._lock:
                                if gid not in self._inbox:
                                    logger.info(f"Adding job {gid} to the inbox")
                                    self._inbox[gid] = job
                    except Exception as e:
                        logger.error(e)

        thread = threading.Thread(target=process, args=())
        thread.daemon = True
        thread.start()

    def execute(self):
        while True:
            try:
                self._load_jobs()
                break
            except Exception as e:
                self._logger.error("Failed to load jobs", e)

        self._start_listener()

        self._logger.info("Processing jobs...")

        # Main loop
        while True:
            # for _ in range(2):
            try:
                # Load inbox to in-progress jobs
                with self._lock:
                    updated_gid = []
                    for gid in self._inbox:
                        self._in_progress_jobs[gid] = self._inbox[gid]
                        updated_gid.append(gid)
                    for gid in updated_gid:
                        del self._inbox[gid]

                jobs_to_remove = []
                if len(self._in_progress_jobs) == 0:
                    self._logger.info("No job to process. Sleep...")
                    time.sleep(2)

                # Process in-progress jobs
                for gid, job in self._in_progress_jobs.items():
                    logger = self._logger.getChild(gid)

                    # If pickup a new job, update the job status to be in-progress
                    if job.get("status") == JobStatus.Pending:
                        logger.info(f"Update job {gid} to {JobStatus.InProgress}")
                        job["status"] = JobStatus.InProgress
                        job["logs"] = []
                        job["outputs"] = []
                        self._update_job(gid, job)

                    logs = job["logs"]
                    # XXX: Remove
                    print(
                        "debug memory",
                        len(logs),
                        id(logs),
                        id(self._in_progress_jobs[gid]["logs"]),
                    )
                    step, status = self._resolve_step(logs)

                    # TODO: Refactor this, a lot of duplicate code
                    logger.info(
                        f"Processing job step: {step.name}, status: {status.name}"
                    )
                    if step == ProcessingStep.ControlNet:
                        if status == ProcessingStatus.Pending:
                            prediction = ControlNetPrediction.populate_from_b64(
                                job["data"], job["word"]
                            )
                            logs.append(prediction)
                        elif status == ProcessingStatus.Processing:
                            prediction = logs[-1]
                            prediction.fetch_log()
                        elif status == ProcessingStatus.Done:
                            prediction = logs[-1]
                            output: PredictionOutput = prediction.get_output()
                            job["outputs"].append(output)
                            step = ProcessingStep.BgRemoval
                            status = ProcessingStatus.Pending
                    if step == ProcessingStep.BgRemoval:
                        if status == ProcessingStatus.Pending:
                            file_path = job["outputs"][-1].img_src
                            prediction = BgRemovalPrediction.populate_from_file(
                                file_path
                            )
                            logs.append(prediction)
                            if prediction.status == PredictionStatus.Finished:
                                status = ProcessingStatus.Done
                        if status == ProcessingStatus.Done:
                            output = prediction.get_output()
                            job["outputs"].append(output)
                            step = ProcessingStep.CRM
                            status = ProcessingStatus.Pending
                    if step == ProcessingStep.CRM:
                        if status == ProcessingStatus.Pending:
                            outputs: list[PredictionOutput] = job["outputs"]
                            b64 = outputs[-1].img_src
                            prediction = CRMPrediction.populate_from_b64(b64)
                            logs.append(prediction)
                        elif status == ProcessingStatus.Processing:
                            prediction = logs[-1]
                            prediction.fetch_log()
                        elif status == ProcessingStatus.Done:
                            prediction = logs[-1]
                            output: PredictionOutput = prediction.get_output()
                            job["outputs"].append(output)
                            job["status"] = JobStatus.Done

                    # Update data to redis
                    updated_job = {}
                    updated_job["status"] = job["status"]
                    updated_job["word"] = job["word"]
                    updated_job["data"] = job["data"]
                    updated_job["logs"] = [
                        {"model": pred.name, "id": pred.id, "status": pred.status}
                        for pred in job["logs"]
                        if isinstance(pred, Prediction)
                    ]
                    updated_job["outputs"] = [
                        asdict(x)
                        for x in job["outputs"]
                        if isinstance(x, PredictionOutput)
                    ]
                    self._update_job(gid, updated_job)

                    if job["status"] == JobStatus.Done:
                        jobs_to_remove.append(gid)

                for gid in jobs_to_remove:
                    del self._in_progress_jobs[gid]

            except Exception as e:
                self._logger.error("Failed to process job", e)

            # XXX: Remove this
            # finally:
            #     break

    def _resolve_step(
        self, logs: list[Prediction]
    ) -> Tuple[ProcessingStep, ProcessingStatus]:
        # XXX: Remove
        print(
            "debug resolve",
            logs,
        )
        prediction = logs[-1] if len(logs) > 0 else None
        model_name = prediction.name if prediction is not None else None
        # XXX: Remove
        print(
            "debug resolve 2",
            prediction,
            logs,
            prediction.status if prediction is not None else None,
        )

        # Resolve step
        step = None
        if model_name is None:
            step = ProcessingStep.ControlNet
        elif model_name == CONTROLNET_NAME:
            step = ProcessingStep.ControlNet
        elif model_name == BGREM_NAME:
            step = ProcessingStep.BgRemoval
        elif model_name == CRM_NAME:
            step = ProcessingStep.CRM
        else:
            raise Exception("Invalid model name")

        # Resolve processing status
        processing_status = None
        if prediction is None:
            processing_status = ProcessingStatus.Pending
        elif prediction.status in [
            PredictionStatus.Starting,
            PredictionStatus.Processing,
        ]:
            processing_status = ProcessingStatus.Processing
        elif prediction.status == PredictionStatus.Finished:
            processing_status = ProcessingStatus.Done
        else:
            # If failed reassign to pending to reprocess
            processing_status = ProcessingStatus.Pending

        return step, processing_status

    def _get_job(self, gid):
        return json.loads(cast(str, self._r.get(gid)))

    def _update_job(self, gid, job):
        self._r.set(gid, json.dumps(job))


if __name__ == "__main__":
    executor = JobExecutor()
    executor.execute()
