"""
    Pipeline Operations

    Pipeline operations are the principle driving force for this project. Each implementation of `PipelineOp` is a modular,
    reusable algorithm which performs a single operation on an image. `PipelineOp` has a simple interface with
    only 3 steps to satisfy the contract:

      1. Declare a constructor with inputs necessary to perform the operation in `#perform`.

      2. Implement `#output` which returns the result of performing the operation in `#perform`.

      3. Implement `#perform` ensuring that the last line is `return self`.
         This provides support to perform the op and immediately assign `#output`
         to local variables.

"""

class PipelineOp:
	def perform(self):
		raise NotImplementedError

	def output(self):
		"""
		Returns the result from performing this operation.
		"""
		raise NotImplementedError