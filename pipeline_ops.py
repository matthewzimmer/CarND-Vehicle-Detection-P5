"""
    Pipeline Operations

    Pipeline operations are the principle driving force for this project. Each implementation of `PipelineOp` is a modular,
    reusable algorithm which performs a single operation on an image. `PipelineOp` has a simple interface with
    only 3 steps to satisfy the contract:

      1. Declare a constructor with inputs necessary to perform the operation in `#perform`.

      2. Implement `#perform`

          * This method must return `self`. This provides support to perform the op and 
            immediately assign the call to `#output` to local variables.

          * Declared your op's output by calling `#_apply_output` once you've performed your operation.

"""


class PipelineOp:
  def __init__(self):
    self.__output = None

  def perform(self):
    raise NotImplementedError

  def output(self):
    return self.__output

  def _apply_output(self, value):
    self.__output = value
    return self