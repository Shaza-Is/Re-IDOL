from pydantic import BaseModel, ValidationError, validator


class TrainArgs(object):
    model: str
    