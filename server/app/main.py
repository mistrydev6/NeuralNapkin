from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from job import JobStatus, add_job, get_job

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Submission(BaseModel):
    base64: str
    word: str
    gid: str


@app.post("/submit")
async def submit(submission: Submission):
    add_job(submission.gid, submission.base64, submission.word)


@app.get("/check/{gid}")
async def check(gid: str):
    job = get_job(gid)
    status = job.get("status")
    return status if status is not None else JobStatus.Pending


@app.get("/output/{gid}")
async def get_output(gid: str):
    job = get_job(gid)
    if job.get("status") != JobStatus.Done:
        return Response(status_code=400)
    output = job["outputs"][-1]
    return FileResponse(
        output["img_src"], media_type="application/octet-stream", filename=f"{gid}.obj"
    )
