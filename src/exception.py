import sys
import logging
from src.logger import logging


def error_message_detail(error, error_detail : sys):
    """
    Generates a detailed error message with the file name, line number, 
    and error description for a given exception.

    Args:
        error (Exception): The exception instance containing the error message.
        error_detail (module): Typically the 'sys' module, used to retrieve 
                               traceback details.

    Returns:
        str: A formatted string containing the file name, line number, and 
             error message, which helps identify where the error occurred.
    """
    _, _, exc_tb  = error_detail.exc_info()
    file_name     = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message

class CustomException(Exception):
    """
    Extends the base Exception class to provide detailed error messages, including file name and line number.

    Args:
        error_message (str): A description of the error.
        error_detail (module): Typically the 'sys' module, used to extract 
                               traceback information for detailed error context.

    Attributes:
        error_message (str): A formatted string containing the file name, line 
                             number, and error message, generated by the 
                             error_message_detail function.

    Methods:
        __str__: Returns the formatted error message for display or logging.
    """
    def __init__(self, error_message, error_detail : sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message