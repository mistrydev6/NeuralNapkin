from abc import ABC
from dataclasses import asdict, dataclass
import os
from tempfile import tempdir
import requests
from typing import Any, Literal, Optional, Union, cast
import numpy as np
import replicate
import cv2
import base64
from rembg import remove
import shortuuid

replicate.default_client._api_token = ""  # type: ignore

CONTROLNET_NAME = "controlnet"
BGREM_NAME = "bgrem"
CRM_NAME = "crm"

OUTPUT_DIR = "./outputs"


class PredictionStatus:
    Starting = "starting"
    Processing = "processing"
    Finished = "finished"
    Failed = "failed"


type ImagePath = str
type B64 = str


@dataclass
class PredictionOutput:
    img_src: Union[ImagePath, B64]
    src_type: Literal["image_path", "b64"]

    # def __init__(
    #     self, img_src: Union[ImagePath, B64], src_type: Literal["image_path", "b64"]
    # ):
    #     self.img_src = img_src
    #     self.src_type = src_type
    def __json__(self):
        return asdict(self)

    @classmethod
    def from_image_path(cls, path):
        return cls(path, "image_path")

    @classmethod
    def from_b64(cls, src):
        return cls(src, "b64")


class Prediction(ABC):
    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name
        self.status = PredictionStatus.Starting

    @classmethod
    def populate_from_b64(cls, b64: str, prompt: str = ""):
        raise NotImplementedError()

    @classmethod
    def populate_from_file(cls, file_path: str, prompt: str = ""):
        raise NotImplementedError()

    def get_output(self) -> Optional[PredictionOutput]:
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        pass

    def fetch_log(self) -> Any:
        pass

    def update_log(self, log: dict):
        self.log = log

    def is_finished(self):
        return self.status == PredictionStatus.Finished

    def is_failed(self):
        return self.status == PredictionStatus.Failed

    def is_pending(self):
        return self.status == PredictionStatus.Starting

    def is_processing(self):
        return self.status == PredictionStatus.Processing


class ReplicatePrediction(Prediction):
    def __init__(self, id: str, name: str, output_extension: str) -> None:
        super().__init__(id, name)
        self.output_extension = output_extension

    def fetch_log(self):
        log = replicate.predictions.get(self.id)
        self.log = log
        status = log.status
        # XXX: Remove
        print("Debug status", status)
        if status == "starting":
            self.status = PredictionStatus.Starting
        elif status == "processing":
            self.status = PredictionStatus.Processing
        elif status == "succeeded":
            self.status = PredictionStatus.Finished
        else:
            self.status = PredictionStatus.Failed
        return log

    def get_output(self) -> Optional[PredictionOutput]:
        super().get_output()
        if self.is_finished() and self.log is not None:
            urls = self.log.output
            url = urls if type(urls) is str else cast(list[str], urls)[-1]
            res = requests.get(url, stream=True)
            file_path = os.path.join(
                OUTPUT_DIR, f"{self.name}_{self.id}.{self.output_extension}"
            )
            with open(file_path, "wb") as file:
                for chunk in res.iter_content(chunk_size=1024):
                    file.write(chunk)
            return PredictionOutput.from_image_path(file_path)


class ControlNetPrediction(ReplicatePrediction):
    def __init__(self, id: str) -> None:
        super().__init__(id, CONTROLNET_NAME, "jpg")

    @classmethod
    def populate_from_b64(cls, b64: str, prompt: str = ""):
        # TODO: Refactor this. Duplicate code with CRM prediction
        input = {
            "image": f"data:application/octet-stream;base64,{b64}",
            # TODO: Prompt engineering
            "prompt": prompt,
        }
        log = replicate.predictions.create(
            version="435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117",
            input=input,
        )
        ins = cls(log.id)
        ins.update_log(log)
        return ins


class CRMPrediction(ReplicatePrediction):
    def __init__(self, id: str) -> None:
        super().__init__(id, CRM_NAME, "obj")

    @classmethod
    def populate_from_b64(cls, b64: str, prompt: str = ""):
        # TODO: Refactor this. Duplicate code with CRM prediction
        input = {
            "image_path": f"data:application/octet-stream;base64,{b64}",
        }
        log = replicate.predictions.create(
            version="9a17c48c7e8a6c1b75788b454a234c13833277afafda546e014e59c3837f8932",
            input=input,
        )
        ins = cls(log.id)
        ins.update_log(log)
        return ins


class BgRemovalPrediction(Prediction):
    def __init__(self, id: str) -> None:
        super().__init__(id, BGREM_NAME)

    @classmethod
    def populate_from_file(cls, file_path: str, prompt: str = ""):
        img = cv2.imread(file_path)
        output = cast(np.ndarray, remove(img))
        _, encoded_img = cv2.imencode(".jpg", output)
        b64_img = base64.b64encode(encoded_img).decode("utf-8")
        log = {"output": b64_img}
        ins = cls(shortuuid.uuid())
        ins.log = log
        ins.status = PredictionStatus.Finished
        return ins

    def get_output(self) -> Optional[PredictionOutput]:
        if self.status == PredictionStatus.Finished:
            return PredictionOutput.from_b64(self.log["output"])


if __name__ == "__main__":
    # out = replicate.predictions.get("x0qvr18vtxrj60cpf3nr85j25m")
    # print(dir(out))
    # print(out.output)
    # res = requests.get(out.output, stream=True)
    # with open("output2.obj", "wb") as file:
    #     for chunk in res.iter_content(chunk_size=1024):
    #         file.write(chunk)

    img = cv2.imread("./dog.jpg")
    _, encoded_img = cv2.imencode(".jpg", img)
    b64_img = base64.b64encode(encoded_img).decode("utf-8")
    input = {
        "image": f"data:application/octet-stream;base64,{b64_img}",
        "prompt": "turtle",
    }
    prediction = replicate.predictions.create(
        version="9a17c48c7e8a6c1b75788b454a234c13833277afafda546e014e59c3837f8932",
        input=input,
    )
    print(prediction)
    # prediction = BgRemovalPrediction.populate_from_file("app/output_1.png")
    # cv2.imwrite("bgrm.jpg", prediction)
