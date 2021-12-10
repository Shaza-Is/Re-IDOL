from pydantic import BaseModel, ValidationError, validator

from app.resources.constants import (
    COMMON_ARGS_VALIDATION_ERROR_INCORRECT_NUMBER, 
    COMMON_ARGS_VALIDATION_ERROR_NOT_INT, 
    COMMON_ARGS_VALIDATION_ERROR_NOT_A_NUMBER
)

class CommonArgs(BaseModel):
    """CommonArgs is a pydantic model. This is used
    to validate common args received by train and test. 

    Args:
        BaseModel : Pydantic base model type

    Raises:
        ValidationError: this will be raised if the option provided is
        not a number, is a float or is a number that is less than 1 
        or greater than 3. 
    """
    option: int
    

    @validator("option")
    def validate_option(cls, value: str):
        """validate_option makes sure that the option provided is from 1 to 3, everything
        else is considered invalid.

        Args:
            value (str): the building option
        """
        try: 
            numerical_value = int(value)
            
            if numerical_value > 3 or numerical_value < 1: 
                raise ValueError(COMMON_ARGS_VALIDATION_ERROR_INCORRECT_NUMBER)

            if not isinstance(numerical_value, int): 
                raise ValueError(COMMON_ARGS_VALIDATION_ERROR_NOT_INT)

            return numerical_value
        except ValueError as error:
            raise ValueError(COMMON_ARGS_VALIDATION_ERROR_NOT_A_NUMBER)

