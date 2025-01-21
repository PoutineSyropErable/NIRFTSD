"""
type(b.localForm()) = <class 'petsc4py.PETSc._Vec_LocalForm'>

b.localForm() =
<petsc4py.PETSc._Vec_LocalForm object at 0x7c63b4923d30>

Help on _Vec_LocalForm object:

class _Vec_LocalForm(builtins.object)
 |  Methods defined here:
 |
 |  __enter__(...)
 |      _Vec_LocalForm.__enter__(self)
 |      Source code at petsc4py/PETSc/petscvec.pxi:631
 |
 |  __exit__(...)
 |      _Vec_LocalForm.__exit__(self, *exc)
 |      Source code at petsc4py/PETSc/petscvec.pxi:636
 |
 |  __init__(self, /, *args, **kwargs)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |
 |  __new__(*args, **kwargs) from builtins.type
 |      Create and return a new object.  See help(type) for accurate signature.



"""
type(loc_b) = <class 'petsc4py.PETSc.Vec'>

loc_b = 
<petsc4py.PETSc.Vec object at 0x757fdd3ea570>

help(loc_b) = 
"""
Help on Vec object:

class Vec(Object)
 |  A vector object.
 |  
 |  See Also
 |  --------
 |  petsc.Vec
 |  
 |  Method resolution order:
 |      Vec
 |      Object
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __abs__(self, /)
 |      abs(self)
 |  
 |  __add__(self, value, /)
 |      Return self+value.
 |  
 |  __delitem__(self, key, /)
 |      Delete self[key].
 |  
 |  __dlpack__(...)
 |      Vec.__dlpack__(self, stream=-1)
 |      Source code at petsc4py/PETSc/Vec.pyx:719
 |  
 |  __dlpack_device__(...)
 |      Vec.__dlpack_device__(self)
 |      Source code at petsc4py/PETSc/Vec.pyx:722
 |  
 |  __enter__(...)
 |      Vec.__enter__(self)
 |      Source code at petsc4py/PETSc/Vec.pyx:132
 |  
 |  __exit__(...)
 |      Vec.__exit__(self, *exc)
 |      Source code at petsc4py/PETSc/Vec.pyx:137
 |  
 |  __getitem__(self, key, /)
 |      Return self[key].
 |  
 |  __iadd__(self, value, /)
 |      Return self+=value.
 |  
 |  __imul__(self, value, /)
 |      Return self*=value.
 |  
 |  __isub__(self, value, /)
 |      Return self-=value.
 |  
 |  __itruediv__(self, value, /)
 |      Return self/=value.
 |  
 |  __matmul__(self, value, /)
 |      Return self@value.
 |  
 |  __mul__(self, value, /)
 |      Return self*value.
 |  
 |  __neg__(self, /)
 |      -self
 |  
 |  __pos__(self, /)
 |      +self
 |  
 |  __radd__(...)
 |  
 |  __rmatmul__(self, value, /)
 |      Return value@self.
 |  
 |  __rmul__(...)
 |  
 |  __rsub__(...)
 |  
 |  __rtruediv__(...)
 |  
 |  __setitem__(self, key, value, /)
 |      Set self[key] to value.
 |  
 |  __sub__(self, value, /)
 |      Return self-value.
 |  
 |  __truediv__(self, value, /)
 |      Return self/value.
 |  
 |  abs(...)
 |      Vec.abs(self) -> None
 |      Replace each entry (xₙ) in the vector by abs|xₙ|.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecAbs
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2293
 |  
 |  appendOptionsPrefix(...)
 |      Vec.appendOptionsPrefix(self, prefix: str | None) -> None
 |      Append to the prefix used for searching for options in the database.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setOptionsPrefix, petsc.VecAppendOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1028
 |  
 |  assemble(...)
 |      Vec.assemble(self) -> None
 |      Assemble the vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              assemblyBegin, assemblyEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3047
 |  
 |  assemblyBegin(...)
 |      Vec.assemblyBegin(self) -> None
 |      Begin an assembling stage of the vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              assemblyEnd, petsc.VecAssemblyBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3023
 |  
 |  assemblyEnd(...)
 |      Vec.assemblyEnd(self) -> None
 |      Finish the assembling stage initiated with `assemblyBegin`.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              assemblyBegin, petsc.VecAssemblyEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3035
 |  
 |  attachDLPackInfo(...)
 |      Vec.attachDLPackInfo(self, vec: Vec | None = None, dltensor=None) -> Self
 |      Attach tensor information from another vector or DLPack tensor.
 |      
 |              Logically collective.
 |      
 |              This tensor information is required when converting a `Vec` to a
 |              DLPack object.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector with attached tensor information. This is typically created
 |                  by calling `createWithDLPack`.
 |              dltensor
 |                  DLPack tensor. This will only be used if ``vec`` is `None`.
 |      
 |              Notes
 |              -----
 |              This operation does not copy any data from ``vec`` or ``dltensor``.
 |      
 |              See Also
 |              --------
 |              clearDLPackInfo, createWithDLPack
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:643
 |  
 |  axpby(...)
 |      Vec.axpby(self, alpha: Scalar, beta: Scalar, x: Vec) -> None
 |      Compute and store y = ɑ·x + β·y.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  First scale factor.
 |              beta
 |                  Second scale factor.
 |              x
 |                  Input vector, must not be the current vector.
 |      
 |              See Also
 |              --------
 |              axpy, aypx, waxpy, petsc.VecAXPBY
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2524
 |  
 |  axpy(...)
 |      Vec.axpy(self, alpha: Scalar, x: Vec) -> None
 |      Compute and store y = ɑ·x + y.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  Scale factor.
 |              x
 |                  Input vector.
 |      
 |              See Also
 |              --------
 |              isaxpy, petsc.VecAXPY
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2460
 |  
 |  aypx(...)
 |      Vec.aypx(self, alpha: Scalar, x: Vec) -> None
 |      Compute and store y = x + ɑ·y.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  Scale factor.
 |              x
 |                  Input vector, must not be the current vector.
 |      
 |              See Also
 |              --------
 |              axpy, axpby, petsc.VecAYPX
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2504
 |  
 |  bindToCPU(...)
 |      Vec.bindToCPU(self, flg: bool) -> None
 |      Bind vector operations execution on the CPU.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              boundToCPU, petsc.VecBindToCPU
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1396
 |  
 |  boundToCPU(...)
 |      Vec.boundToCPU(self) -> bool
 |      Return whether the vector has been bound to the CPU.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              bindToCPU, petsc.VecBoundToCPU
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1409
 |  
 |  chop(...)
 |      Vec.chop(self, tol: float) -> None
 |      Set all vector entries less than some absolute tolerance to zero.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              tol
 |                  The absolute tolerance below which entries are set to zero.
 |      
 |              See Also
 |              --------
 |              petsc.VecFilter
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1732
 |  
 |  clearDLPackInfo(...)
 |      Vec.clearDLPackInfo(self) -> Self
 |      Clear tensor information.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              attachDLPackInfo, createWithDLPack
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:705
 |  
 |  conjugate(...)
 |      Vec.conjugate(self) -> None
 |      Conjugate the vector.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecConjugate
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2305
 |  
 |  copy(...)
 |      Vec.copy(self, result: Vec | None = None) -> Vec
 |      Return a copy of the vector.
 |      
 |              Logically collective.
 |      
 |              This operation copies vector entries to the new vector.
 |      
 |              Parameters
 |              ----------
 |              result
 |                  Target vector for the copy. If `None` then a new vector is
 |                  created internally.
 |      
 |              See Also
 |              --------
 |              duplicate, petsc.VecCopy
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1707
 |  
 |  create(...)
 |      Vec.create(self, comm: Comm | None = None) -> Self
 |      Create a vector object.
 |      
 |              Collective.
 |      
 |              After creation the vector type can then be set with `setType`.
 |      
 |              Parameters
 |              ----------
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              destroy, petsc.VecCreate
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:176
 |  
 |  createCUDAWithArrays(...)
 |      Vec.createCUDAWithArrays(self, cpuarray: Sequence[Scalar] | None = None, cudahandle: Any | None = None, size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a `Type.CUDA` vector with optional arrays.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              cpuarray
 |                  Host array. Will be lazily allocated if not provided.
 |              cudahandle
 |                  Address of the array on the GPU. Will be lazily allocated if
 |                  not provided.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateSeqCUDAWithArrays, petsc.VecCreateMPICUDAWithArrays
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:370
 |  
 |  createGhost(...)
 |      Vec.createGhost(self, ghosts: Sequence[int], size: LayoutSizeSpec, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a parallel vector with ghost padding on each processor.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              ghosts
 |                  Global indices of ghost points.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              createGhostWithArray, petsc.VecCreateGhost, petsc.VecCreateGhostBlock
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:819
 |  
 |  createGhostWithArray(...)
 |      Vec.createGhostWithArray(self, ghosts: Sequence[int], array: Sequence[Scalar], size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a parallel vector with ghost padding and provided arrays.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              ghosts
 |                  Global indices of ghost points.
 |              array
 |                  Array to store the vector values. Must be at least as large as
 |                  the local size of the vector (including ghost points).
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              createGhost, petsc.VecCreateGhostWithArray
 |              petsc.VecCreateGhostBlockWithArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:861
 |  
 |  createHIPWithArrays(...)
 |      Vec.createHIPWithArrays(self, cpuarray: Sequence[Scalar] | None = None, hiphandle: Any | None = None, size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a `Type.HIP` vector with optional arrays.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              cpuarray
 |                  Host array. Will be lazily allocated if not provided.
 |              hiphandle
 |                  Address of the array on the GPU. Will be lazily allocated if
 |                  not provided.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateSeqHIPWithArrays, petsc.VecCreateMPIHIPWithArrays
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:428
 |  
 |  createLocalVector(...)
 |      Vec.createLocalVector(self) -> Vec
 |      Create a local vector.
 |      
 |              Not collective.
 |      
 |              Returns
 |              -------
 |              Vec
 |                  The local vector.
 |      
 |              See Also
 |              --------
 |              getLocalVector, petsc.VecCreateLocalVector
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1204
 |  
 |  createMPI(...)
 |      Vec.createMPI(self, size: LayoutSizeSpec, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a parallel `Type.MPI` vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              createSeq, petsc.VecCreateMPI
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:283
 |  
 |  createNest(...)
 |      Vec.createNest(self, vecs: Sequence[Vec], isets: Sequence[IS] = None, comm: Comm | None = None) -> Self
 |      Create a `Type.NEST` vector containing multiple nested subvectors.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Iterable of subvectors.
 |              isets
 |                  Iterable of index sets for each nested subvector.
 |                  Defaults to contiguous ordering.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateNest
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:952
 |  
 |  createSeq(...)
 |      Vec.createSeq(self, size: LayoutSizeSpec, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a sequential `Type.SEQ` vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `COMM_SELF`.
 |      
 |              See Also
 |              --------
 |              createMPI, petsc.VecCreateSeq
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:247
 |  
 |  createShared(...)
 |      Vec.createShared(self, size: LayoutSizeSpec, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a `Type.SHARED` vector that uses shared memory.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateShared
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:918
 |  
 |  createViennaCLWithArrays(...)
 |      Vec.createViennaCLWithArrays(self, cpuarray: Sequence[Scalar] | None = None, viennaclvechandle: Any | None = None, size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a `Type.VIENNACL` vector with optional arrays.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              cpuarray
 |                  Host array. Will be lazily allocated if not provided.
 |              viennaclvechandle
 |                  Address of the array on the GPU. Will be lazily allocated if
 |                  not provided.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateSeqViennaCLWithArrays
 |              petsc.VecCreateMPIViennaCLWithArrays
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:486
 |  
 |  createWithArray(...)
 |      Vec.createWithArray(self, array: Sequence[Scalar], size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a vector using a provided array.
 |      
 |              Collective.
 |      
 |              This method will create either a `Type.SEQ` or `Type.MPI`
 |              depending on the size of the communicator.
 |      
 |              Parameters
 |              ----------
 |              array
 |                  Array to store the vector values. Must be at least as large as
 |                  the local size of the vector.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc.VecCreateSeqWithArray, petsc.VecCreateMPIWithArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:319
 |  
 |  createWithDLPack(...)
 |      Vec.createWithDLPack(self, dltensor, size: LayoutSizeSpec | None = None, bsize: int | None = None, comm: Comm | None = None) -> Self
 |      Create a vector wrapping a DLPack object, sharing the same memory.
 |      
 |              Collective.
 |      
 |              This operation does not modify the storage of the original tensor and
 |              should be used with contiguous tensors only. If the tensor is stored in
 |              row-major order (e.g. PyTorch tensors), the resulting vector will look
 |              like an unrolled tensor using row-major order.
 |      
 |              The resulting vector type will be one of `Type.SEQ`, `Type.MPI`,
 |              `Type.SEQCUDA`, `Type.MPICUDA`, `Type.SEQHIP` or
 |              `Type.MPIHIP` depending on the type of ``dltensor`` and the number
 |              of processes in the communicator.
 |      
 |              Parameters
 |              ----------
 |              dltensor
 |                  Either an object with a ``__dlpack__`` method or a DLPack tensor object.
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:545
 |  
 |  destroy(...)
 |      Vec.destroy(self) -> Self
 |      Destroy the vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              create, petsc.VecDestroy
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:163
 |  
 |  dot(...)
 |      Vec.dot(self, vec: Vec) -> Scalar
 |      Return the dot product with ``vec``.
 |      
 |              Collective.
 |      
 |              For complex numbers this computes yᴴ·x with ``self`` as x, ``vec``
 |              as y and where yᴴ denotes the conjugate transpose of y.
 |      
 |              Use `tDot` for the indefinite form yᵀ·x where yᵀ denotes the
 |              transpose of y.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector to compute the dot product with.
 |      
 |              See Also
 |              --------
 |              dotBegin, dotEnd, tDot, petsc.VecDot
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1787
 |  
 |  dotBegin(...)
 |      Vec.dotBegin(self, vec: Vec) -> None
 |      Begin computing the dot product.
 |      
 |              Collective.
 |      
 |              This should be paired with a call to `dotEnd`.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector to compute the dot product with.
 |      
 |              See Also
 |              --------
 |              dotEnd, dot, petsc.VecDotBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1812
 |  
 |  dotEnd(...)
 |      Vec.dotEnd(self, vec: Vec) -> Scalar
 |      Finish computing the dot product initiated with `dotBegin`.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              dotBegin, dot, petsc.VecDotEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1832
 |  
 |  dotNorm2(...)
 |      Vec.dotNorm2(self, vec: Vec) -> tuple[Scalar, float]
 |      Return the dot product with ``vec`` and its squared norm.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              dot, norm, petsc.VecDotNorm2
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2151
 |  
 |  duplicate(...)
 |      Vec.duplicate(self, array: Sequence[Scalar] | None = None) -> Vec
 |      Create a new vector with the same type, optionally with data.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              array
 |                  Optional values to store in the new vector.
 |      
 |              See Also
 |              --------
 |              copy, petsc.VecDuplicate
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1682
 |  
 |  equal(...)
 |      Vec.equal(self, vec: Vec) -> bool
 |      Return whether the vector is equal to another.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector to compare with.
 |      
 |              See Also
 |              --------
 |              petsc.VecEqual
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1768
 |  
 |  exp(...)
 |      Vec.exp(self) -> None
 |      Replace each entry (xₙ) in the vector by exp(xₙ).
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              log, petsc.VecExp
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2257
 |  
 |  getArray(...)
 |      Vec.getArray(self, readonly: bool = False) -> ArrayScalar
 |      Return local portion of the vector as an `ndarray`.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              readonly
 |                  Request read-only access.
 |      
 |              See Also
 |              --------
 |              setArray, getBuffer
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1313
 |  
 |  getBlockSize(...)
 |      Vec.getBlockSize(self) -> int
 |      Return the block size of the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecGetBlockSize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1149
 |  
 |  getBuffer(...)
 |      Vec.getBuffer(self, readonly: bool = False) -> Any
 |      Return a buffered view of the local portion of the vector.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              readonly
 |                  Request read-only access.
 |      
 |              Returns
 |              -------
 |              typing.Any
 |                  `Buffer object <python:c-api/buffer>` wrapping the local portion of
 |                  the vector data. This can be used either as a context manager
 |                  providing access as a numpy array or can be passed to array
 |                  constructors accepting buffered objects such as `numpy.asarray`.
 |      
 |              Examples
 |              --------
 |              Accessing the data with a context manager:
 |      
 |              >>> vec = PETSc.Vec().createWithArray([1, 2, 3])
 |              >>> with vec.getBuffer() as arr:
 |              ...     arr
 |              array([1., 2., 3.])
 |      
 |              Converting the buffer to an `ndarray`:
 |      
 |              >>> buf = PETSc.Vec().createWithArray([1, 2, 3]).getBuffer()
 |              >>> np.asarray(buf)
 |              array([1., 2., 3.])
 |      
 |              See Also
 |              --------
 |              getArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1270
 |  
 |  getCLContextHandle(...)
 |      Vec.getCLContextHandle(self) -> int
 |      Return the OpenCL context associated with the vector.
 |      
 |              Not collective.
 |      
 |              Returns
 |              -------
 |              int
 |                  Pointer to underlying CL context. This can be used with
 |                  `pyopencl` through `pyopencl.Context.from_int_ptr`.
 |      
 |              See Also
 |              --------
 |              getCLQueueHandle, petsc.VecViennaCLGetCLContext
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1593
 |  
 |  getCLMemHandle(...)
 |      Vec.getCLMemHandle(self, mode: AccessModeSpec = 'rw') -> int
 |      Return the OpenCL buffer associated with the vector.
 |      
 |              Not collective.
 |      
 |              Returns
 |              -------
 |              int
 |                  Pointer to the device buffer. This can be used with
 |                  `pyopencl` through `pyopencl.Context.from_int_ptr`.
 |      
 |              Notes
 |              -----
 |              This method may incur a host-to-device copy if the device data is
 |              out of date and ``mode`` is ``"r"`` or ``"rw"``.
 |      
 |              See Also
 |              --------
 |              restoreCLMemHandle, petsc.VecViennaCLGetCLMem
 |              petsc.VecViennaCLGetCLMemRead, petsc.VecViennaCLGetCLMemWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1633
 |  
 |  getCLQueueHandle(...)
 |      Vec.getCLQueueHandle(self) -> int
 |      Return the OpenCL command queue associated with the vector.
 |      
 |              Not collective.
 |      
 |              Returns
 |              -------
 |              int
 |                  Pointer to underlying CL command queue. This can be used with
 |                  `pyopencl` through `pyopencl.Context.from_int_ptr`.
 |      
 |              See Also
 |              --------
 |              getCLContextHandle, petsc.VecViennaCLGetCLQueue
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1613
 |  
 |  getCUDAHandle(...)
 |      Vec.getCUDAHandle(self, mode: AccessModeSpec = 'rw') -> Any
 |      Return a pointer to the device buffer.
 |      
 |              Not collective.
 |      
 |              The returned pointer should be released using `restoreCUDAHandle`
 |              with the same access mode.
 |      
 |              Returns
 |              -------
 |              typing.Any
 |                  CUDA device pointer.
 |      
 |              Notes
 |              -----
 |              This method may incur a host-to-device copy if the device data is
 |              out of date and ``mode`` is ``"r"`` or ``"rw"``.
 |      
 |              See Also
 |              --------
 |              restoreCUDAHandle, petsc.VecCUDAGetArray, petsc.VecCUDAGetArrayRead
 |              petsc.VecCUDAGetArrayWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1423
 |  
 |  getDM(...)
 |      Vec.getDM(self) -> DM
 |      Return the `DM` associated to the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setDM, petsc.VecGetDM
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3519
 |  
 |  getGhostIS(...)
 |      Vec.getGhostIS(self) -> IS
 |      Return ghosting indices of a ghost vector.
 |      
 |              Collective.
 |      
 |              Returns
 |              -------
 |              IS
 |                  Indices of ghosts.
 |      
 |              See Also
 |              --------
 |              petsc.VecGhostGetGhostIS
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3385
 |  
 |  getHIPHandle(...)
 |      Vec.getHIPHandle(self, mode: AccessModeSpec = 'rw') -> Any
 |      Return a pointer to the device buffer.
 |      
 |              Not collective.
 |      
 |              The returned pointer should be released using `restoreHIPHandle`
 |              with the same access mode.
 |      
 |              Returns
 |              -------
 |              typing.Any
 |                  HIP device pointer.
 |      
 |              Notes
 |              -----
 |              This method may incur a host-to-device copy if the device data is
 |              out of date and ``mode`` is ``"r"`` or ``"rw"``.
 |      
 |              See Also
 |              --------
 |              restoreHIPHandle, petsc.VecHIPGetArray, petsc.VecHIPGetArrayRead
 |              petsc.VecHIPGetArrayWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1495
 |  
 |  getLGMap(...)
 |      Vec.getLGMap(self) -> LGMap
 |      Return the local-to-global mapping.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setLGMap, petsc.VecGetLocalToGlobalMapping
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2899
 |  
 |  getLocalSize(...)
 |      Vec.getLocalSize(self) -> int
 |      Return the local size of the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setSizes, getSize, petsc.VecGetLocalSize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1107
 |  
 |  getLocalVector(...)
 |      Vec.getLocalVector(self, lvec: Vec, readonly: bool = False) -> None
 |      Maps the local portion of the vector into a local vector.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              lvec
 |                  The local vector obtained from `createLocalVector`.
 |              readonly
 |                  Request read-only access.
 |      
 |              See Also
 |              --------
 |              createLocalVector, restoreLocalVector, petsc.VecGetLocalVectorRead
 |              petsc.VecGetLocalVector
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1223
 |  
 |  getNestSubVecs(...)
 |      Vec.getNestSubVecs(self) -> list[Vec]
 |      Return all the vectors contained in the nested vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setNestSubVecs, petsc.VecNestGetSubVecs
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3451
 |  
 |  getOffloadMask(...)
 |      Vec.getOffloadMask(self) -> int
 |      Return the offloading status of the vector.
 |      
 |              Not collective.
 |      
 |              Common return values include:
 |      
 |              - 1: ``PETSC_OFFLOAD_CPU`` - CPU has valid entries
 |              - 2: ``PETSC_OFFLOAD_GPU`` - GPU has valid entries
 |              - 3: ``PETSC_OFFLOAD_BOTH`` - CPU and GPU are in sync
 |      
 |              Returns
 |              -------
 |              int
 |                  Enum value from `petsc.PetscOffloadMask` describing the offloading
 |                  status.
 |      
 |              See Also
 |              --------
 |              petsc.VecGetOffloadMask, petsc.PetscOffloadMask
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1567
 |  
 |  getOptionsPrefix(...)
 |      Vec.getOptionsPrefix(self) -> str
 |      Return the prefix used for searching for options in the database.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setOptionsPrefix, petsc.VecGetOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1014
 |  
 |  getOwnershipRange(...)
 |      Vec.getOwnershipRange(self) -> tuple[int, int]
 |      Return the locally owned range of indices ``(start, end)``.
 |      
 |              Not collective.
 |      
 |              Returns
 |              -------
 |              start : int
 |                  The first local element.
 |              end : int
 |                  One more than the last local element.
 |      
 |              See Also
 |              --------
 |              getOwnershipRanges, petsc.VecGetOwnershipRange
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1163
 |  
 |  getOwnershipRanges(...)
 |      Vec.getOwnershipRanges(self) -> ArrayInt
 |      Return the range of indices owned by each process.
 |      
 |              Not collective.
 |      
 |              The returned array is the result of exclusive scan of the local sizes.
 |      
 |              See Also
 |              --------
 |              getOwnershipRange, petsc.VecGetOwnershipRanges
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1184
 |  
 |  getSize(...)
 |      Vec.getSize(self) -> int
 |      Return the global size of the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setSizes, getLocalSize, petsc.VecGetSize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1093
 |  
 |  getSizes(...)
 |      Vec.getSizes(self) -> LayoutSizeSpec
 |      Return the vector sizes.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              getSize, getLocalSize, petsc.VecGetLocalSize, petsc.VecGetSize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1121
 |  
 |  getSubVector(...)
 |      Vec.getSubVector(self, iset: IS, subvec: Vec | None = None) -> Vec
 |      Return a subvector from given indices.
 |      
 |              Collective.
 |      
 |              Once finished with the subvector it should be returned with
 |              `restoreSubVector`.
 |      
 |              Parameters
 |              ----------
 |              iset
 |                  Index set describing which indices to extract into the subvector.
 |              subvec
 |                  Subvector to copy entries into. If `None` then a new `Vec` will
 |                  be created.
 |      
 |              See Also
 |              --------
 |              restoreSubVector, petsc.VecGetSubVector
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3406
 |  
 |  getType(...)
 |      Vec.getType(self) -> str
 |      Return the type of the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setType, petsc.VecGetType
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1079
 |  
 |  getValue(...)
 |      Vec.getValue(self, index: int) -> Scalar
 |      Return a single value from the vector.
 |      
 |              Not collective.
 |      
 |              Only values locally stored may be accessed.
 |      
 |              Parameters
 |              ----------
 |              index
 |                  Location of the value to read.
 |      
 |              See Also
 |              --------
 |              getValues, petsc.VecGetValues
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2720
 |  
 |  getValues(...)
 |      Vec.getValues(self, indices: Sequence[int], values: Sequence[Scalar] | None = None) -> ArrayScalar
 |      Return values from certain locations in the vector.
 |      
 |              Not collective.
 |      
 |              Only values locally stored may be accessed.
 |      
 |              Parameters
 |              ----------
 |              indices
 |                  Locations of the values to read.
 |              values
 |                  Location to store the collected values. If not provided then a new
 |                  array will be allocated.
 |      
 |              See Also
 |              --------
 |              getValue, setValues, petsc.VecGetValues
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2742
 |  
 |  getValuesStagStencil(...)
 |      Vec.getValuesStagStencil(self, indices, values=None) -> None
 |      Not implemented.
 |      Source code at petsc4py/PETSc/Vec.pyx:2767
 |  
 |  ghostUpdate(...)
 |      Vec.ghostUpdate(self, addv: InsertModeSpec = None, mode: ScatterModeSpec = None) -> None
 |      Update ghosted vector entries.
 |      
 |              Neighborwise collective.
 |      
 |              Parameters
 |              ----------
 |              addv
 |                  Insertion mode.
 |              mode
 |                  Scatter mode.
 |      
 |              Examples
 |              --------
 |              To accumulate ghost region values onto owning processes:
 |      
 |              >>> vec.ghostUpdate(InsertMode.ADD_VALUES, ScatterMode.REVERSE)
 |      
 |              Update ghost regions:
 |      
 |              >>> vec.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)
 |      
 |              See Also
 |              --------
 |              ghostUpdateBegin, ghostUpdateEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3331
 |  
 |  ghostUpdateBegin(...)
 |      Vec.ghostUpdateBegin(self, addv: InsertModeSpec = None, mode: ScatterModeSpec = None) -> None
 |      Begin updating ghosted vector entries.
 |      
 |              Neighborwise collective.
 |      
 |              See Also
 |              --------
 |              ghostUpdateEnd, ghostUpdate, createGhost, petsc.VecGhostUpdateBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3297
 |  
 |  ghostUpdateEnd(...)
 |      Vec.ghostUpdateEnd(self, addv: InsertModeSpec = None, mode: ScatterModeSpec = None) -> None
 |      Finish updating ghosted vector entries initiated with `ghostUpdateBegin`.
 |      
 |              Neighborwise collective.
 |      
 |              See Also
 |              --------
 |              ghostUpdateBegin, ghostUpdate, createGhost, petsc.VecGhostUpdateEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3314
 |  
 |  isaxpy(...)
 |      Vec.isaxpy(self, idx: IS, alpha: Scalar, x: Vec) -> None
 |      Add a scaled reduced-space vector to a subset of the vector.
 |      
 |              Logically collective.
 |      
 |              This is equivalent to ``y[idx[i]] += alpha*x[i]``.
 |      
 |              Parameters
 |              ----------
 |              idx
 |                  Index set for the reduced space. Negative indices are skipped.
 |              alpha
 |                  Scale factor.
 |              x
 |                  Reduced-space vector.
 |      
 |              See Also
 |              --------
 |              axpy, aypx, axpby, petsc.VecISAXPY
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2480
 |  
 |  isset(...)
 |      Vec.isset(self, idx: IS, alpha: Scalar) -> None
 |      Set specific elements of the vector to the same value.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              idx
 |                  Index set specifying the vector entries to set.
 |              alpha
 |                  Value to set the selected entries to.
 |      
 |              See Also
 |              --------
 |              set, zeroEntries, petsc.VecISSet
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2383
 |  
 |  load(...)
 |      Vec.load(self, viewer: Viewer) -> Self
 |      Load a vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              view, petsc.VecLoad
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1750
 |  
 |  localForm(...)
 |      Vec.localForm(self) -> Any
 |      Return a context manager for viewing ghost vectors in local form.
 |      
 |              Logically collective.
 |      
 |              Returns
 |              -------
 |              typing.Any
 |                  Context manager yielding the vector in local (ghosted) form.
 |      
 |              Notes
 |              -----
 |              This operation does not perform a copy. To obtain up-to-date ghost
 |              values `ghostUpdateBegin` and `ghostUpdateEnd` must be called
 |              first.
 |      
 |              Non-ghost values can be found
 |              at ``values[0:nlocal]`` and ghost values at
 |              ``values[nlocal:nlocal+nghost]``.
 |      
 |              Examples
 |              --------
 |              >>> with vec.localForm() as lf:
 |              ...     # compute with lf
 |      
 |              See Also
 |              --------
 |              createGhost, ghostUpdateBegin, ghostUpdateEnd
 |              petsc.VecGhostGetLocalForm, petsc.VecGhostRestoreLocalForm
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3264
 |  
 |  log(...)
 |      Vec.log(self) -> None
 |      Replace each entry in the vector by its natural logarithm.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              exp, petsc.VecLog
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2269
 |  
 |  mDot(...)
 |      Vec.mDot(self, vecs: Sequence[Vec], out: ArrayScalar | None = None) -> ArrayScalar
 |      Compute Xᴴ·y with X an array of vectors.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Optional placeholder for the result.
 |      
 |              See Also
 |              --------
 |              dot, tDot, mDotBegin, mDotEnd, petsc.VecMDot
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1902
 |  
 |  mDotBegin(...)
 |      Vec.mDotBegin(self, vecs: Sequence[Vec], out: ArrayScalar) -> None
 |      Starts a split phase multiple dot product computation.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Placeholder for the result.
 |      
 |              See Also
 |              --------
 |              mDot, mDotEnd, petsc.VecMDotBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1935
 |  
 |  mDotEnd(...)
 |      Vec.mDotEnd(self, vecs: Sequence[Vec], out: ArrayScalar) -> ArrayScalar
 |      Ends a split phase multiple dot product computation.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Placeholder for the result.
 |      
 |              See Also
 |              --------
 |              mDot, mDotBegin, petsc.VecMDotEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1965
 |  
 |  max(...)
 |      Vec.max(self) -> tuple[int, float]
 |      Return the vector entry with maximum real part and its location.
 |      
 |              Collective.
 |      
 |              Returns
 |              -------
 |              p : int
 |                  Location of the maximum value. If multiple entries exist with the
 |                  same value then the smallest index will be returned.
 |              val : Scalar
 |                  Minimum value.
 |      
 |              See Also
 |              --------
 |              min, petsc.VecMax
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2203
 |  
 |  maxPointwiseDivide(...)
 |      Vec.maxPointwiseDivide(self, vec: Vec) -> float
 |      Return the maximum of the component-wise absolute value division.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``result = max_i abs(x[i] / y[i])``.
 |      
 |              Parameters
 |              ----------
 |              x
 |                  Numerator vector.
 |              y
 |                  Denominator vector.
 |      
 |              See Also
 |              --------
 |              pointwiseMin, pointwiseMax, pointwiseMaxAbs
 |              petsc.VecMaxPointwiseDivide
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2696
 |  
 |  maxpy(...)
 |      Vec.maxpy(self, alphas: Sequence[Scalar], vecs: Sequence[Vec]) -> None
 |      Compute and store y = Σₙ(ɑₙ·Xₙ) + y with X an array of vectors.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``y[:] = alphas[i]*vecs[i, :] + y[:]``.
 |      
 |              Parameters
 |              ----------
 |              alphas
 |                  Array of scale factors, one for each vector in ``vecs``.
 |              vecs
 |                  Array of vectors.
 |      
 |              See Also
 |              --------
 |              axpy, aypx, axpby, waxpy, petsc.VecMAXPY
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2569
 |  
 |  min(...)
 |      Vec.min(self) -> tuple[int, float]
 |      Return the vector entry with minimum real part and its location.
 |      
 |              Collective.
 |      
 |              Returns
 |              -------
 |              p : int
 |                  Location of the minimum value. If multiple entries exist with the
 |                  same value then the smallest index will be returned.
 |              val : Scalar
 |                  Minimum value.
 |      
 |              See Also
 |              --------
 |              max, petsc.VecMin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2180
 |  
 |  mtDot(...)
 |      Vec.mtDot(self, vecs: Sequence[Vec], out: ArrayScalar | None = None) -> ArrayScalar
 |      Compute Xᵀ·y with X an array of vectors.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Optional placeholder for the result.
 |      
 |              See Also
 |              --------
 |              tDot, mDot, mtDotBegin, mtDotEnd, petsc.VecMTDot
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1996
 |  
 |  mtDotBegin(...)
 |      Vec.mtDotBegin(self, vecs: Sequence[Vec], out: ArrayScalar) -> None
 |      Starts a split phase transpose multiple dot product computation.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Placeholder for the result.
 |      
 |              See Also
 |              --------
 |              mtDot, mtDotEnd, petsc.VecMTDotBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2029
 |  
 |  mtDotEnd(...)
 |      Vec.mtDotEnd(self, vecs: Sequence[Vec], out: ArrayScalar) -> ArrayScalar
 |      Ends a split phase transpose multiple dot product computation.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  Array of vectors.
 |              out
 |                  Placeholder for the result.
 |      
 |              See Also
 |              --------
 |              mtDot, mtDotBegin, petsc.VecMTDotEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2059
 |  
 |  norm(...)
 |      Vec.norm(self, norm_type: NormTypeSpec = None) -> float | tuple[float, float]
 |      Compute the vector norm.
 |      
 |              Collective.
 |      
 |              A 2-tuple is returned if `NormType.NORM_1_AND_2` is specified.
 |      
 |              See Also
 |              --------
 |              petsc.VecNorm, petsc.NormType
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2090
 |  
 |  normBegin(...)
 |      Vec.normBegin(self, norm_type: NormTypeSpec = None) -> None
 |      Begin computing the vector norm.
 |      
 |              Collective.
 |      
 |              This should be paired with a call to `normEnd`.
 |      
 |              See Also
 |              --------
 |              normEnd, norm, petsc.VecNormBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2112
 |  
 |  normEnd(...)
 |      Vec.normEnd(self, norm_type: NormTypeSpec = None) -> float | tuple[float, float]
 |      Finish computations initiated with `normBegin`.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              normBegin, norm, petsc.VecNormEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2131
 |  
 |  normalize(...)
 |      Vec.normalize(self) -> float
 |      Normalize the vector by its 2-norm.
 |      
 |              Collective.
 |      
 |              Returns
 |              -------
 |              float
 |                  The vector norm before normalization.
 |      
 |              See Also
 |              --------
 |              norm, petsc.VecNormalize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2226
 |  
 |  permute(...)
 |      Vec.permute(self, order: IS, invert: bool = False) -> None
 |      Permute the vector in-place with a provided ordering.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              order
 |                  Ordering for the permutation.
 |              invert
 |                  Whether to invert the permutation.
 |      
 |              See Also
 |              --------
 |              petsc.VecPermute
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2337
 |  
 |  placeArray(...)
 |      Vec.placeArray(self, array: Sequence[Scalar]) -> None
 |      Set the local portion of the vector to a provided array.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              resetArray, setArray, petsc.VecPlaceArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1345
 |  
 |  pointwiseDivide(...)
 |      Vec.pointwiseDivide(self, x: Vec, y: Vec) -> None
 |      Compute and store the component-wise division of two vectors.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``w[i] = x[i] / y[i]``.
 |      
 |              Parameters
 |              ----------
 |              x
 |                  Numerator vector.
 |              y
 |                  Denominator vector.
 |      
 |              See Also
 |              --------
 |              pointwiseMult, petsc.VecPointwiseDivide
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2618
 |  
 |  pointwiseMax(...)
 |      Vec.pointwiseMax(self, x: Vec, y: Vec) -> None
 |      Compute and store the component-wise maximum of two vectors.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``w[i] = max(x[i], y[i])``.
 |      
 |              Parameters
 |              ----------
 |              x, y
 |                  Input vectors to find the component-wise maxima.
 |      
 |              See Also
 |              --------
 |              pointwiseMin, pointwiseMaxAbs, petsc.VecPointwiseMax
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2658
 |  
 |  pointwiseMaxAbs(...)
 |      Vec.pointwiseMaxAbs(self, x: Vec, y: Vec) -> None
 |      Compute and store the component-wise maximum absolute values.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``w[i] = max(abs(x[i]), abs(y[i]))``.
 |      
 |              Parameters
 |              ----------
 |              x, y
 |                  Input vectors to find the component-wise maxima.
 |      
 |              See Also
 |              --------
 |              pointwiseMin, pointwiseMax, petsc.VecPointwiseMaxAbs
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2677
 |  
 |  pointwiseMin(...)
 |      Vec.pointwiseMin(self, x: Vec, y: Vec) -> None
 |      Compute and store the component-wise minimum of two vectors.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``w[i] = min(x[i], y[i])``.
 |      
 |              Parameters
 |              ----------
 |              x, y
 |                  Input vectors to find the component-wise minima.
 |      
 |              See Also
 |              --------
 |              pointwiseMax, pointwiseMaxAbs, petsc.VecPointwiseMin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2639
 |  
 |  pointwiseMult(...)
 |      Vec.pointwiseMult(self, x: Vec, y: Vec) -> None
 |      Compute and store the component-wise multiplication of two vectors.
 |      
 |              Logically collective.
 |      
 |              Equivalent to ``w[i] = x[i] * y[i]``.
 |      
 |              Parameters
 |              ----------
 |              x, y
 |                  Input vectors to multiply component-wise.
 |      
 |              See Also
 |              --------
 |              pointwiseDivide, petsc.VecPointwiseMult
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2599
 |  
 |  reciprocal(...)
 |      Vec.reciprocal(self) -> None
 |      Replace each entry in the vector by its reciprocal.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecReciprocal
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2245
 |  
 |  resetArray(...)
 |      Vec.resetArray(self, force: bool = False) -> ArrayScalar | None
 |      Reset the vector to use its default array.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              force
 |                  Force the calling of `petsc.VecResetArray` even if no user array
 |                  has been placed with `placeArray`.
 |      
 |              Returns
 |              -------
 |              ArrayScalar
 |                  The array previously provided by the user with `placeArray`.
 |                  Can be `None` if ``force`` is `True` and no array was placed
 |                  before.
 |      
 |              See Also
 |              --------
 |              placeArray, petsc.VecResetArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1366
 |  
 |  restoreCLMemHandle(...)
 |      Vec.restoreCLMemHandle(self) -> None
 |      Restore a pointer to the OpenCL buffer obtained with `getCLMemHandle`.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              getCLMemHandle, petsc.VecViennaCLRestoreCLMemWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1670
 |  
 |  restoreCUDAHandle(...)
 |      Vec.restoreCUDAHandle(self, handle: Any, mode: AccessModeSpec = 'rw') -> None
 |      Restore a pointer to the device buffer obtained with `getCUDAHandle`.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              handle
 |                  CUDA device pointer.
 |              mode
 |                  Access mode.
 |      
 |              See Also
 |              --------
 |              getCUDAHandle, petsc.VecCUDARestoreArray
 |              petsc.VecCUDARestoreArrayRead, petsc.VecCUDARestoreArrayWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1462
 |  
 |  restoreHIPHandle(...)
 |      Vec.restoreHIPHandle(self, handle: Any, mode: AccessModeSpec = 'rw') -> None
 |      Restore a pointer to the device buffer obtained with `getHIPHandle`.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              handle
 |                  HIP device pointer.
 |              mode
 |                  Access mode.
 |      
 |              See Also
 |              --------
 |              getHIPHandle, petsc.VecHIPRestoreArray, petsc.VecHIPRestoreArrayRead
 |              petsc.VecHIPRestoreArrayWrite
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1534
 |  
 |  restoreLocalVector(...)
 |      Vec.restoreLocalVector(self, lvec: Vec, readonly: bool = False) -> None
 |      Unmap a local access obtained with `getLocalVector`.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              lvec
 |                  The local vector.
 |              readonly
 |                  Request read-only access.
 |      
 |              See Also
 |              --------
 |              createLocalVector, getLocalVector, petsc.VecRestoreLocalVectorRead
 |              petsc.VecRestoreLocalVector
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1246
 |  
 |  restoreSubVector(...)
 |      Vec.restoreSubVector(self, iset: IS, subvec: Vec) -> None
 |      Restore a subvector extracted using `getSubVector`.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              iset
 |                  Index set describing the indices represented by the subvector.
 |              subvec
 |                  Subvector to be restored.
 |      
 |              See Also
 |              --------
 |              getSubVector, petsc.VecRestoreSubVector
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3432
 |  
 |  scale(...)
 |      Vec.scale(self, alpha: Scalar) -> None
 |      Scale all entries of the vector.
 |      
 |              Collective.
 |      
 |              This method sets each entry (xₙ) in the vector to ɑ·xₙ.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  The scaling factor.
 |      
 |              See Also
 |              --------
 |              shift, petsc.VecScale
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2403
 |  
 |  set(...)
 |      Vec.set(self, alpha: Scalar) -> None
 |      Set all components of the vector to the same value.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              zeroEntries, isset, petsc.VecSet
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2370
 |  
 |  setArray(...)
 |      Vec.setArray(self, array: Sequence[Scalar]) -> None
 |      Set values for the local portion of the vector.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              placeArray
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1333
 |  
 |  setBlockSize(...)
 |      Vec.setBlockSize(self, bsize: int) -> None
 |      Set the block size of the vector.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecSetBlockSize
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1136
 |  
 |  setDM(...)
 |      Vec.setDM(self, dm: DM) -> None
 |      Associate a `DM` to the vector.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              getDM, petsc.VecSetDM
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3507
 |  
 |  setFromOptions(...)
 |      Vec.setFromOptions(self) -> None
 |      Configure the vector from the options database.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, petsc.VecSetFromOptions
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1042
 |  
 |  setLGMap(...)
 |      Vec.setLGMap(self, lgmap: LGMap) -> None
 |      Set the local-to-global mapping.
 |      
 |              Logically collective.
 |      
 |              This allows users to insert vector entries using a local numbering
 |              with `setValuesLocal`.
 |      
 |              See Also
 |              --------
 |              setValues, setValuesLocal, getLGMap, petsc.VecSetLocalToGlobalMapping
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2884
 |  
 |  setMPIGhost(...)
 |      Vec.setMPIGhost(self, ghosts: Sequence[int]) -> None
 |      Set the ghost points for a ghosted vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              ghosts
 |                  Global indices of ghost points.
 |      
 |              See Also
 |              --------
 |              createGhost
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3366
 |  
 |  setNestSubVecs(...)
 |      Vec.setNestSubVecs(self, sx: Sequence[Vec], idxm: Sequence[int] | None = None) -> None
 |      Set the component vectors at specified indices in the nested vector.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              sx
 |                  Array of component vectors.
 |              idxm
 |                  Indices of the component vectors, defaults to ``range(len(sx))``.
 |      
 |              See Also
 |              --------
 |              getNestSubVecs, petsc.VecNestSetSubVecs
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3473
 |  
 |  setOption(...)
 |      Vec.setOption(self, option: Option, flag: bool) -> None
 |      Set option.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecSetOption
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1067
 |  
 |  setOptionsPrefix(...)
 |      Vec.setOptionsPrefix(self, prefix: str | None) -> None
 |      Set the prefix used for searching for options in the database.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, getOptionsPrefix, petsc.VecSetOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1000
 |  
 |  setRandom(...)
 |      Vec.setRandom(self, random: Random | None = None) -> None
 |      Set all components of the vector to random numbers.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              random
 |                  Random number generator. If `None` then one will be created
 |                  internally.
 |      
 |              See Also
 |              --------
 |              petsc.VecSetRandom
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2317
 |  
 |  setSizes(...)
 |      Vec.setSizes(self, size: LayoutSizeSpec, bsize: int | None = None) -> None
 |      Set the local and global sizes of the vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              size
 |                  Vector size.
 |              bsize
 |                  Vector block size. If `None`, ``bsize = 1``.
 |      
 |              See Also
 |              --------
 |              getSizes, petsc.VecSetSizes
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:218
 |  
 |  setType(...)
 |      Vec.setType(self, vec_type: Type | str) -> None
 |      Set the vector type.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vec_type
 |                  The vector type.
 |      
 |              See Also
 |              --------
 |              create, getType, petsc.VecSetType
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:199
 |  
 |  setUp(...)
 |      Vec.setUp(self) -> Self
 |      Set up the internal data structures for using the vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              create, destroy, petsc.VecSetUp
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1054
 |  
 |  setValue(...)
 |      Vec.setValue(self, index: int, value: Scalar, addv: InsertModeSpec = None) -> None
 |      Insert or add a single value in the vector.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              index
 |                  Location to write to. Negative indices are ignored.
 |              value
 |                  Value to insert at ``index``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValue` cannot be made with different values
 |              for ``addv`` without intermediate calls to `assemblyBegin` and
 |              `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValues, petsc.VecSetValues
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2771
 |  
 |  setValueLocal(...)
 |      Vec.setValueLocal(self, index: int, value: Scalar, addv: InsertModeSpec = None) -> None
 |      Insert or add a single value in the vector using a local numbering.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              index
 |                  Location to write to.
 |              value
 |                  Value to insert at ``index``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValueLocal` cannot be made with different
 |              values for ``addv`` without intermediate calls to `assemblyBegin`
 |              and `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValuesLocal, petsc.VecSetValuesLocal
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2914
 |  
 |  setValues(...)
 |      Vec.setValues(self, indices: Sequence[int], values: Sequence[Scalar], addv: InsertModeSpec = None) -> None
 |      Insert or add multiple values in the vector.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              indices
 |                  Locations to write to. Negative indices are ignored.
 |              values
 |                  Values to insert at ``indices``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValues` cannot be made with different values
 |              for ``addv`` without intermediate calls to `assemblyBegin` and
 |              `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValue, petsc.VecSetValues
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2808
 |  
 |  setValuesBlocked(...)
 |      Vec.setValuesBlocked(self, indices: Sequence[int], values: Sequence[Scalar], addv: InsertModeSpec = None) -> None
 |      Insert or add blocks of values in the vector.
 |      
 |              Not collective.
 |      
 |              Equivalent to ``x[bs*indices[i]+j] = y[bs*i+j]`` for
 |              ``0 <= i < len(indices)``, ``0 <= j < bs`` and ``bs`` `block_size`.
 |      
 |              Parameters
 |              ----------
 |              indices
 |                  Block indices to write to. Negative indices are ignored.
 |              values
 |                  Values to insert at ``indices``. Should have length
 |                  ``len(indices) * vec.block_size``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValuesBlocked` cannot be made with different
 |              values for ``addv`` without intermediate calls to `assemblyBegin`
 |              and `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValues, petsc.VecSetValuesBlocked
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2842
 |  
 |  setValuesBlockedLocal(...)
 |      Vec.setValuesBlockedLocal(self, indices: Sequence[int], values: Sequence[Scalar], addv: InsertModeSpec = None) -> None
 |      Insert or add blocks of values in the vector with a local numbering.
 |      
 |              Not collective.
 |      
 |              Equivalent to ``x[bs*indices[i]+j] = y[bs*i+j]`` for
 |              ``0 <= i < len(indices)``, ``0 <= j < bs`` and ``bs`` `block_size`.
 |      
 |              Parameters
 |              ----------
 |              indices
 |                  Local block indices to write to.
 |              values
 |                  Values to insert at ``indices``. Should have length
 |                  ``len(indices) * vec.block_size``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValuesBlockedLocal` cannot be made with
 |              different values for ``addv`` without intermediate calls to
 |              `assemblyBegin` and `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValuesBlocked, setValuesLocal, petsc.VecSetValuesBlockedLocal
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2985
 |  
 |  setValuesLocal(...)
 |      Vec.setValuesLocal(self, indices: Sequence[int], values: Sequence[Scalar], addv: InsertModeSpec = None) -> None
 |      Insert or add multiple values in the vector with a local numbering.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              indices
 |                  Locations to write to.
 |              values
 |                  Values to insert at ``indices``.
 |              addv
 |                  Insertion mode.
 |      
 |              Notes
 |              -----
 |              The values may be cached so `assemblyBegin` and `assemblyEnd`
 |              must be called after all calls of this method are completed.
 |      
 |              Multiple calls to `setValuesLocal` cannot be made with different
 |              values for ``addv`` without intermediate calls to `assemblyBegin`
 |              and `assemblyEnd`.
 |      
 |              See Also
 |              --------
 |              setValues, petsc.VecSetValuesLocal
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2951
 |  
 |  setValuesStagStencil(...)
 |      Vec.setValuesStagStencil(self, indices, values, addv=None) -> None
 |      Not implemented.
 |      Source code at petsc4py/PETSc/Vec.pyx:2880
 |  
 |  shift(...)
 |      Vec.shift(self, alpha: Scalar) -> None
 |      Shift all entries in the vector.
 |      
 |              Collective.
 |      
 |              This method sets each entry (xₙ) in the vector to xₙ + ɑ.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  The shift to apply to the vector values.
 |      
 |              See Also
 |              --------
 |              scale, petsc.VecShift
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2423
 |  
 |  sqrtabs(...)
 |      Vec.sqrtabs(self) -> None
 |      Replace each entry (xₙ) in the vector by √|xₙ|.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecSqrtAbs
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2281
 |  
 |  strideGather(...)
 |      Vec.strideGather(self, field: int, vec: Vec, addv: InsertModeSpec = None) -> None
 |      Insert component values into a single-component vector.
 |      
 |              Collective.
 |      
 |              The current vector is expected to be multi-component (`block_size`
 |              greater than ``1``) and the target vector is expected to be
 |              single-component.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |              vec
 |                  Single-component vector to be inserted into.
 |              addv
 |                  Insertion mode.
 |      
 |              See Also
 |              --------
 |              strideScatter, petsc.VecStrideScatter
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3231
 |  
 |  strideMax(...)
 |      Vec.strideMax(self, field: int) -> tuple[int, float]
 |      Return the maximum of entries in a subvector.
 |      
 |              Collective.
 |      
 |              Equivalent to ``max(x[field], x[field+bs], x[field+2*bs], ...)`` where
 |              ``bs`` is `block_size`.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |      
 |              Returns
 |              -------
 |              int
 |                  Location of maximum.
 |              float
 |                  Maximum value.
 |      
 |              See Also
 |              --------
 |              strideScale, strideSum, strideMin, petsc.VecStrideMax
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3137
 |  
 |  strideMin(...)
 |      Vec.strideMin(self, field: int) -> tuple[int, float]
 |      Return the minimum of entries in a subvector.
 |      
 |              Collective.
 |      
 |              Equivalent to ``min(x[field], x[field+bs], x[field+2*bs], ...)`` where
 |              ``bs`` is `block_size`.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |      
 |              Returns
 |              -------
 |              int
 |                  Location of minimum.
 |              float
 |                  Minimum value.
 |      
 |              See Also
 |              --------
 |              strideScale, strideSum, strideMax, petsc.VecStrideMin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3106
 |  
 |  strideNorm(...)
 |      Vec.strideNorm(self, field: int, norm_type: NormTypeSpec = None) -> float | tuple[float, float]
 |      Return the norm of entries in a subvector.
 |      
 |              Collective.
 |      
 |              Equivalent to ``norm(x[field], x[field+bs], x[field+2*bs], ...)`` where
 |              ``bs`` is `block_size`.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |              norm_type
 |                  The norm type.
 |      
 |              See Also
 |              --------
 |              norm, strideScale, strideSum, petsc.VecStrideNorm
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3168
 |  
 |  strideScale(...)
 |      Vec.strideScale(self, field: int, alpha: Scalar) -> None
 |      Scale a component of the vector.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |              alpha
 |                  Factor to multiple the component entries by.
 |      
 |              See Also
 |              --------
 |              strideSum, strideMin, strideMax, petsc.VecStrideScale
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3062
 |  
 |  strideScatter(...)
 |      Vec.strideScatter(self, field: int, vec: Vec, addv: InsertModeSpec = None) -> None
 |      Scatter entries into a component of another vector.
 |      
 |              Collective.
 |      
 |              The current vector is expected to be single-component
 |              (`block_size` of ``1``) and the target vector is expected to be
 |              multi-component.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |              vec
 |                  Multi-component vector to be scattered into.
 |              addv
 |                  Insertion mode.
 |      
 |              See Also
 |              --------
 |              strideGather, petsc.VecStrideScatter
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3200
 |  
 |  strideSum(...)
 |      Vec.strideSum(self, field: int) -> Scalar
 |      Sum subvector entries.
 |      
 |              Collective.
 |      
 |              Equivalent to ``sum(x[field], x[field+bs], x[field+2*bs], ...)`` where
 |              ``bs`` is `block_size`.
 |      
 |              Parameters
 |              ----------
 |              field
 |                  Component index. Must be between ``0`` and ``vec.block_size``.
 |      
 |              See Also
 |              --------
 |              strideScale, strideMin, strideMax, petsc.VecStrideSum
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3083
 |  
 |  sum(...)
 |      Vec.sum(self) -> Scalar
 |      Return the sum of all the entries of the vector.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc.VecSum
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2166
 |  
 |  swap(...)
 |      Vec.swap(self, vec: Vec) -> None
 |      Swap the content of two vectors.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  The vector to swap data with.
 |      
 |              See Also
 |              --------
 |              petsc.VecSwap
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2443
 |  
 |  tDot(...)
 |      Vec.tDot(self, vec: Vec) -> Scalar
 |      Return the indefinite dot product with ``vec``.
 |      
 |              Collective.
 |      
 |              This computes yᵀ·x with ``self`` as x, ``vec``
 |              as y and where yᵀ denotes the transpose of y.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector to compute the indefinite dot product with.
 |      
 |              See Also
 |              --------
 |              tDotBegin, tDotEnd, dot, petsc.VecTDot
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1846
 |  
 |  tDotBegin(...)
 |      Vec.tDotBegin(self, vec: Vec) -> None
 |      Begin computing the indefinite dot product.
 |      
 |              Collective.
 |      
 |              This should be paired with a call to `tDotEnd`.
 |      
 |              Parameters
 |              ----------
 |              vec
 |                  Vector to compute the indefinite dot product with.
 |      
 |              See Also
 |              --------
 |              tDotEnd, tDot, petsc.VecTDotBegin
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1868
 |  
 |  tDotEnd(...)
 |      Vec.tDotEnd(self, vec: Vec) -> Scalar
 |      Finish computing the indefinite dot product initiated with `tDotBegin`.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              tDotBegin, tDot, petsc.VecTDotEnd
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:1888
 |  
 |  toDLPack(...)
 |      Vec.toDLPack(self, mode: AccessModeSpec = 'rw') -> Any
 |      Return a DLPack `PyCapsule` wrapping the vector data.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              mode
 |                  Access mode for the vector.
 |      
 |              Returns
 |              -------
 |              `PyCapsule`
 |                  Capsule of a DLPack tensor wrapping a `Vec`.
 |      
 |              Notes
 |              -----
 |              It is important that the access mode is respected by the consumer
 |              as this is not enforced internally.
 |      
 |              See Also
 |              --------
 |              createWithDLPack
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:726
 |  
 |  view(...)
 |      Vec.view(self, viewer: Viewer | None = None) -> None
 |      Display the vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              viewer
 |                  A `Viewer` instance or `None` for the default viewer.
 |      
 |              See Also
 |              --------
 |              load, petsc.VecView
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:144
 |  
 |  waxpy(...)
 |      Vec.waxpy(self, alpha: Scalar, x: Vec, y: Vec) -> None
 |      Compute and store w = ɑ·x + y.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              alpha
 |                  Scale factor.
 |              x
 |                  First input vector.
 |              y
 |                  Second input vector.
 |      
 |              See Also
 |              --------
 |              axpy, aypx, axpby, maxpy, petsc.VecWAXPY
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2547
 |  
 |  zeroEntries(...)
 |      Vec.zeroEntries(self) -> None
 |      Set all entries in the vector to zero.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              set, petsc.VecZeroEntries
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:2358
 |  
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |  
 |  concatenate(...) from builtins.type
 |      Vec.concatenate(cls, vecs: Sequence[Vec]) -> tuple[Vec, list[IS]]
 |      Concatenate vectors into a single vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              vecs
 |                  The vectors to be concatenated.
 |      
 |              Returns
 |              -------
 |              vector_out : Vec
 |                  The concatenated vector.
 |              indices_list : list of IS
 |                  A list of index sets corresponding to the concatenated components.
 |      
 |              See Also
 |              --------
 |              petsc.VecConcatenate
 |      
 |              
 |      Source code at petsc4py/PETSc/Vec.pyx:3535
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  __new__(*args, **kwargs) from builtins.type
 |      Create and return a new object.  See help(type) for accurate signature.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __array_interface__
 |  
 |  array
 |      Vec.array: ArrayScalar
 |      Alias for `array_w`.
 |      Source code at petsc4py/PETSc/Vec.pyx:3641
 |  
 |  array_r
 |      Vec.array_r: ArrayScalar
 |      Read-only `ndarray` containing the local portion of the vector.
 |      Source code at petsc4py/PETSc/Vec.pyx:3631
 |  
 |  array_w
 |      Vec.array_w: ArrayScalar
 |      Writeable `ndarray` containing the local portion of the vector.
 |      Source code at petsc4py/PETSc/Vec.pyx:3622
 |  
 |  block_size
 |      Vec.block_size: int
 |      The block size.
 |      Source code at petsc4py/PETSc/Vec.pyx:3597
 |  
 |  buffer
 |      Vec.buffer: Any
 |      Alias for `buffer_w`.
 |      Source code at petsc4py/PETSc/Vec.pyx:3636
 |  
 |  buffer_r
 |      Vec.buffer_r: Any
 |      Read-only buffered view of the local portion of the vector.
 |      Source code at petsc4py/PETSc/Vec.pyx:3617
 |  
 |  buffer_w
 |      Vec.buffer_w: Any
 |      Writeable buffered view of the local portion of the vector.
 |      Source code at petsc4py/PETSc/Vec.pyx:3612
 |  
 |  local_size
 |      Vec.local_size: int
 |      The local vector size.
 |      Source code at petsc4py/PETSc/Vec.pyx:3592
 |  
 |  owner_range
 |      Vec.owner_range: tuple[int, int]
 |      The locally owned range of indices in the form ``[low, high)``.
 |      Source code at petsc4py/PETSc/Vec.pyx:3602
 |  
 |  owner_ranges
 |      Vec.owner_ranges: ArrayInt
 |      The range of indices owned by each process.
 |      Source code at petsc4py/PETSc/Vec.pyx:3607
 |  
 |  size
 |      Vec.size: int
 |      The global vector size.
 |      Source code at petsc4py/PETSc/Vec.pyx:3587
 |  
 |  sizes
 |      Vec.sizes: LayoutSizeSpec
 |      The local and global vector sizes.
 |      Source code at petsc4py/PETSc/Vec.pyx:3579
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  Option = <class 'petsc4py.PETSc.VecOption'>
 |      Vector assembly option.
 |  
 |  
 |  Type = <class 'petsc4py.PETSc.VecType'>
 |      The vector type.
 |  
 |  
 |  __pyx_vtable__ = <capsule object NULL>
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from Object:
 |  
 |  __bool__(self, /)
 |      True if self else False
 |  
 |  __copy__(...)
 |      Object.__copy__(self)
 |      Source code at petsc4py/PETSc/Object.pyx:32
 |  
 |  __deepcopy__(...)
 |      Object.__deepcopy__(self, memo: dict)
 |      Source code at petsc4py/PETSc/Object.pyx:40
 |  
 |  __eq__(self, value, /)
 |      Return self==value.
 |  
 |  __ge__(self, value, /)
 |      Return self>=value.
 |  
 |  __gt__(self, value, /)
 |      Return self>value.
 |  
 |  __le__(self, value, /)
 |      Return self<=value.
 |  
 |  __lt__(self, value, /)
 |      Return self<value.
 |  
 |  __ne__(self, value, /)
 |      Return self!=value.
 |  
 |  compose(...)
 |      Object.compose(self, name: str | None, obj: Object) -> None
 |      Associate a PETSc object using a key string.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              name
 |                  The string identifying the object to be composed.
 |              obj
 |                  The object to be composed.
 |      
 |              See Also
 |              --------
 |              query, petsc.PetscObjectCompose
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:324
 |  
 |  decRef(...)
 |      Object.decRef(self) -> int
 |      Decrement the object reference count.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              getRefCount, petsc.PetscObjectDereference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:384
 |  
 |  destroyOptionsHandlers(...)
 |      Object.destroyOptionsHandlers(self) -> None
 |      Clear all the option handlers.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setOptionsHandler, petsc.PetscObjectDestroyOptionsHandlers
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:208
 |  
 |  getAttr(...)
 |      Object.getAttr(self, name: str) -> object
 |      Return the attribute associated with a given name.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setAttr, getDict
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:403
 |  
 |  getClassId(...)
 |      Object.getClassId(self) -> int
 |      Return the class identifier of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetClassId
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:265
 |  
 |  getClassName(...)
 |      Object.getClassName(self) -> str
 |      Return the class name of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetClassName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:279
 |  
 |  getComm(...)
 |      Object.getComm(self) -> Comm
 |      Return the communicator of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetComm
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:223
 |  
 |  getDict(...)
 |      Object.getDict(self) -> dict
 |      Return the dictionary of attributes.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setAttr, getAttr
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:431
 |  
 |  getId(...)
 |      Object.getId(self) -> int
 |      Return the unique identifier of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetId
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:308
 |  
 |  getName(...)
 |      Object.getName(self) -> str
 |      Return the name of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:237
 |  
 |  getRefCount(...)
 |      Object.getRefCount(self) -> int
 |      Return the reference count of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetReference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:293
 |  
 |  getTabLevel(...)
 |      Object.getTabLevel(self) -> None
 |      Return the PETSc object tab level.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setTabLevel, incrementTabLevel, petsc.PetscObjectGetTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:513
 |  
 |  incRef(...)
 |      Object.incRef(self) -> int
 |      Increment the object reference count.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              getRefCount, petsc.PetscObjectReference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:367
 |  
 |  incrementTabLevel(...)
 |      Object.incrementTabLevel(self, tab: int, parent: Object | None = None) -> None
 |      Increment the PETSc object tab level.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              setTabLevel, getTabLevel, petsc.PetscObjectIncrementTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:486
 |  
 |  query(...)
 |      Object.query(self, name: str) -> Object
 |      Query for the PETSc object associated with a key string.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              compose, petsc.PetscObjectQuery
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:347
 |  
 |  setAttr(...)
 |      Object.setAttr(self, name: str, attr: object) -> None
 |      Set an the attribute associated with a given name.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              getAttr, getDict
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:417
 |  
 |  setName(...)
 |      Object.setName(self, name: str | None) -> None
 |      Associate a name to the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectSetName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:251
 |  
 |  setOptionsHandler(...)
 |      Object.setOptionsHandler(self, handler: PetscOptionsHandlerFunction | None) -> None
 |      Set the callback for processing extra options.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              handler
 |                  The callback function, called at the end of `setFromOptions`.
 |      
 |              See Also
 |              --------
 |              petsc_options, setFromOptions, petsc.PetscObjectAddOptionsHandler
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:187
 |  
 |  setTabLevel(...)
 |      Object.setTabLevel(self, level: int) -> None
 |      Set the PETSc object tab level.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              incrementTabLevel, getTabLevel, petsc.PetscObjectSetTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:500
 |  
 |  stateGet(...)
 |      Object.stateGet(self) -> int
 |      Return the PETSc object state.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              stateSet, stateIncrease, petsc.PetscObjectStateGet
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:457
 |  
 |  stateIncrease(...)
 |      Object.stateIncrease(self) -> None
 |      Increment the PETSc object state.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              stateGet, stateSet, petsc.PetscObjectStateIncrease
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:445
 |  
 |  stateSet(...)
 |      Object.stateSet(self, state: int) -> None
 |      Set the PETSc object state.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              stateIncrease, stateGet, petsc.PetscObjectStateSet
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:471
 |  
 |  viewFromOptions(...)
 |      Object.viewFromOptions(self, name: str, objpre: Object | None = None) -> None
 |      View the object via command line options.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              name
 |                  The command line option.
 |              objpre
 |                  Optional object that provides prefix.
 |      
 |              See Also
 |              --------
 |              petsc_options, petsc.PetscObjectViewFromOptions
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:164
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from Object:
 |  
 |  classid
 |      Object.classid: int
 |      The class identifier.
 |      Source code at petsc4py/PETSc/Object.pyx:558
 |  
 |  comm
 |      Object.comm: Comm
 |      The object communicator.
 |      Source code at petsc4py/PETSc/Object.pyx:545
 |  
 |  fortran
 |      Object.fortran: int
 |      Fortran handle.
 |      Source code at petsc4py/PETSc/Object.pyx:588
 |  
 |  handle
 |      Object.handle: int
 |      Handle for ctypes support.
 |      Source code at petsc4py/PETSc/Object.pyx:580
 |  
 |  id
 |      Object.id: int
 |      The object identifier.
 |      Source code at petsc4py/PETSc/Object.pyx:563
 |  
 |  klass
 |      Object.klass: str
 |      The class name.
 |      Source code at petsc4py/PETSc/Object.pyx:568
 |  
 |  name
 |      Object.name: str
 |      The object name.
 |      Source code at petsc4py/PETSc/Object.pyx:550
 |  
 |  prefix
 |      Object.prefix: str
 |      Options prefix.
 |      Source code at petsc4py/PETSc/Object.pyx:537
 |  
 |  refcount
 |      Object.refcount: int
 |      Reference count.
 |      Source code at petsc4py/PETSc/Object.pyx:573
 |  
 |  type
 |      Object.type: str
 |      Object type.
 |      Source code at petsc4py/PETSc/Object.pyx:529
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from Object:
 |  
 |  __hash__ = None
"""



type(solver) = <class 'petsc4py.PETSc.KSP'>

solver =  <petsc4py.PETSc.KSP object at 0x7479354e7a10>

help(solver) = 
"""
Help on KSP object:

class KSP(Object)
 |  Abstract PETSc object that manages all Krylov methods.
 |  
 |  This is the object that manages the linear solves in PETSc (even
 |  those such as direct solvers that do no use Krylov accelerators).
 |  
 |  Notes
 |  -----
 |  When a direct solver is used, but no Krylov solver is used, the KSP
 |  object is still used but with a `Type.PREONLY`, meaning that
 |  only application of the preconditioner is used as the linear
 |  solver.
 |  
 |  See Also
 |  --------
 |  create, setType, SNES, TS, PC, Type.CG, Type.GMRES,
 |  petsc.KSP
 |  
 |  Method resolution order:
 |      KSP
 |      Object
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __call__(...)
 |      Solve linear system.
 |      
 |      Collective.
 |      
 |      Parameters
 |      ----------
 |      b
 |          Right hand side vector.
 |      x
 |          Solution vector.
 |      
 |      Notes
 |      -----
 |      Shortcut for `solve`, which returns the solution vector.
 |      
 |      See Also
 |      --------
 |      solve, petsc_options, petsc.KSPSolve
 |  
 |  addConvergenceTest(...)
 |      KSP.addConvergenceTest(self, converged: KSPConvergenceTestFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None, prepend: bool = False) -> None
 |      Add the function to be used to determine convergence.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              converged
 |                  Callback which computes the convergence.
 |              args
 |                  Positional arguments for callback function.
 |              kargs
 |                  Keyword arguments for callback function.
 |              prepend
 |                  Whether to prepend this call before the default
 |                  convergence test or call it after.
 |      
 |              Notes
 |              -----
 |              Cannot be mixed with a call to `setConvergenceTest`.
 |              It can only be called once. If called multiple times, it will
 |              generate an error.
 |      
 |              See Also
 |              --------
 |              setTolerances, getConvergenceTest, setConvergenceTest,
 |              petsc.KSPSetConvergenceTest, petsc.KSPConvergedDefault
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1065
 |  
 |  appendOptionsPrefix(...)
 |      KSP.appendOptionsPrefix(self, prefix: str | None) -> None
 |      Append to prefix used for all `KSP` options in the database.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              prefix
 |                  The options prefix to append.
 |      
 |              Notes
 |              -----
 |              A hyphen (-) must NOT be given at the beginning of the prefix
 |              name. The first character of all runtime options is
 |              AUTOMATICALLY the hyphen.
 |      
 |              See Also
 |              --------
 |              petsc.KSPAppendOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:575
 |  
 |  buildResidual(...)
 |      KSP.buildResidual(self, r: Vec | None = None) -> Vec
 |      Return the residual of the linear system.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              r
 |                  Optional vector to use for the result.
 |      
 |              See Also
 |              --------
 |              buildSolution, petsc.KSPBuildResidual
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2063
 |  
 |  buildSolution(...)
 |      KSP.buildSolution(self, x: Vec | None = None) -> Vec
 |      Return the solution vector.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              x
 |                  Optional vector to store the solution.
 |      
 |              See Also
 |              --------
 |              buildResidual, petsc.KSPBuildSolution
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2041
 |  
 |  callConvergenceTest(...)
 |      KSP.callConvergenceTest(self, its: int, rnorm: float) -> None
 |      Call the convergence test callback.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              its
 |                  Number of iterations.
 |              rnorm
 |                  The residual norm.
 |      
 |              Notes
 |              -----
 |              This functionality is implemented in petsc4py.
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1121
 |  
 |  cancelMonitor = monitorCancel(...)
 |  
 |  computeEigenvalues(...)
 |      KSP.computeEigenvalues(self) -> ArrayComplex
 |      Compute the extreme eigenvalues for the preconditioned operator.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPComputeEigenvalues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2085
 |  
 |  computeExtremeSingularValues(...)
 |      KSP.computeExtremeSingularValues(self) -> tuple[float, float]
 |      Compute the extreme singular values for the preconditioned operator.
 |      
 |              Collective.
 |      
 |              Returns
 |              -------
 |              smax : float
 |                  The maximum singular value.
 |              smin : float
 |                  The minimum singular value.
 |      
 |              See Also
 |              --------
 |              petsc.KSPComputeExtremeSingularValues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2108
 |  
 |  create(...)
 |      KSP.create(self, comm: Comm | None = None) -> Self
 |      Create the KSP context.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPCreate
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:457
 |  
 |  createPython(...)
 |      KSP.createPython(self, context: Any = None, comm: Comm | None = None) -> Self
 |      Create a linear solver of Python type.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              context
 |                  An instance of the Python class implementing the required
 |                  methods.
 |              comm
 |                  MPI communicator, defaults to `Sys.getDefaultComm`.
 |      
 |              See Also
 |              --------
 |              petsc_python_ksp, setType, setPythonContext, Type.PYTHON
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2154
 |  
 |  destroy(...)
 |      KSP.destroy(self) -> Self
 |      Destroy KSP context.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPDestroy
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:444
 |  
 |  getAppCtx(...)
 |      KSP.getAppCtx(self) -> Any
 |      Return the user-defined context for the linear solver.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setAppCtx
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:640
 |  
 |  getCGObjectiveValue(...)
 |      KSP.getCGObjectiveValue(self) -> float
 |      Return the CG objective function value.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPCGGetObjFcn
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1882
 |  
 |  getComputeEigenvalues(...)
 |      KSP.getComputeEigenvalues(self) -> bool
 |      Return flag indicating whether eigenvalues will be calculated.
 |      
 |              Not collective.
 |      
 |              Return the flag indicating that the extreme eigenvalues values
 |              will be calculated via a Lanczos or Arnoldi process as the
 |              linear system is solved.
 |      
 |              See Also
 |              --------
 |              setComputeEigenvalues, petsc.KSPSetComputeEigenvalues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1430
 |  
 |  getComputeSingularValues(...)
 |      KSP.getComputeSingularValues(self) -> bool
 |      Return flag indicating whether singular values will be calculated.
 |      
 |              Not collective.
 |      
 |              Return the flag indicating whether the extreme singular values
 |              will be calculated via a Lanczos or Arnoldi process as the
 |              linear system is solved.
 |      
 |              See Also
 |              --------
 |              setComputeSingularValues, petsc.KSPGetComputeSingularValues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1474
 |  
 |  getConvergedReason(...)
 |      KSP.getConvergedReason(self) -> KSP.ConvergedReason
 |      Use `reason` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1876
 |  
 |  getConvergenceHistory(...)
 |      KSP.getConvergenceHistory(self) -> ArrayReal
 |      Return array containing the residual history.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setConvergenceHistory, petsc.KSPGetResidualHistory
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1188
 |  
 |  getConvergenceTest(...)
 |      KSP.getConvergenceTest(self) -> KSPConvergenceTestFunction
 |      Return the function to be used to determine convergence.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              setTolerances, setConvergenceTest, petsc.KSPGetConvergenceTest
 |              petsc.KSPConvergedDefault
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1108
 |  
 |  getDM(...)
 |      KSP.getDM(self) -> DM
 |      Return the `DM` that may be used by some preconditioners.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, DM, petsc.KSPGetDM
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:654
 |  
 |  getErrorIfNotConverged(...)
 |      KSP.getErrorIfNotConverged(self) -> bool
 |      Return the flag indicating the solver will error if divergent.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetErrorIfNotConverged
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1946
 |  
 |  getHPDDMType(...)
 |      KSP.getHPDDMType(self) -> HPDDMType
 |      Return the Krylov solver type.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPHPDDMGetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1914
 |  
 |  getInitialGuessKnoll(...)
 |      KSP.getInitialGuessKnoll(self) -> bool
 |      Determine whether the KSP solver is using the Knoll trick.
 |      
 |              Not collective.
 |      
 |              This uses the Knoll trick; using `PC.apply` to compute the
 |              initial guess.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetInitialGuessKnoll
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1550
 |  
 |  getInitialGuessNonzero(...)
 |      KSP.getInitialGuessNonzero(self) -> bool
 |      Determine whether the KSP solver uses a zero initial guess.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetInitialGuessNonzero
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1516
 |  
 |  getIterationNumber(...)
 |      KSP.getIterationNumber(self) -> int
 |      Use `its` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1854
 |  
 |  getMonitor(...)
 |      KSP.getMonitor(self) -> KSPMonitorFunction
 |      Return function used to monitor the residual.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setMonitor, monitor, monitorCancel
 |              petsc.KSPGetMonitorContext
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1266
 |  
 |  getNormType(...)
 |      KSP.getNormType(self) -> NormType
 |      Return the norm that is used for convergence testing.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              NormType, setNormType, petsc.KSPGetNormType, petsc.KSPConvergedSkip
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1390
 |  
 |  getOperators(...)
 |      KSP.getOperators(self) -> tuple[Mat, Mat]
 |      Return the matrix associated with the linear system.
 |      
 |              Collective.
 |      
 |              Return the matrix associated with the linear system and a
 |              (possibly) different one used to construct the preconditioner.
 |      
 |              Returns
 |              -------
 |              A : Mat
 |                  Matrix that defines the linear system.
 |              P : Mat
 |                  Matrix to be used in constructing the preconditioner,
 |                  usually the same as ``A``.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, solve, setOperators, petsc.KSPGetOperators
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:850
 |  
 |  getOptionsPrefix(...)
 |      KSP.getOptionsPrefix(self) -> str
 |      Return the prefix used for all `KSP` options in the database.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:561
 |  
 |  getPC(...)
 |      KSP.getPC(self) -> PC
 |      Return the preconditioner.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, setPC, petsc.KSPGetPC
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:897
 |  
 |  getPCSide(...)
 |      KSP.getPCSide(self) -> PC.Side
 |      Return the preconditioning side.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setPCSide, setNormType, getNormType, petsc.KSPGetPCSide
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1349
 |  
 |  getPythonContext(...)
 |      KSP.getPythonContext(self) -> Any
 |      Return the instance of the class implementing Python methods.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_python_ksp, setPythonContext
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2195
 |  
 |  getPythonType(...)
 |      KSP.getPythonType(self) -> str
 |      Return the fully qualified Python name of the class used by the solver.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_python_ksp, setPythonContext, setPythonType
 |              petsc.KSPPythonGetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2225
 |  
 |  getResidualNorm(...)
 |      KSP.getResidualNorm(self) -> float
 |      Use `norm` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1865
 |  
 |  getRhs(...)
 |      KSP.getRhs(self) -> Vec
 |      Return the right-hand side vector for the linear system.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetRhs
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1960
 |  
 |  getSolution(...)
 |      KSP.getSolution(self) -> Vec
 |      Return the solution for the linear system to be solved.
 |      
 |              Not collective.
 |      
 |              Note that this may not be the solution that is stored during
 |              the iterative process.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetSolution
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1975
 |  
 |  getTolerances(...)
 |      KSP.getTolerances(self) -> tuple[float, float, float, int]
 |      Return various tolerances used by the KSP convergence tests.
 |      
 |              Not collective.
 |      
 |              Return the relative, absolute, divergence, and maximum iteration
 |              tolerances used by the default KSP convergence tests.
 |      
 |              Returns
 |              -------
 |              rtol : float
 |                  The relative convergence tolerance
 |              atol : float
 |                  The absolute convergence tolerance
 |              dtol : float
 |                  The divergence tolerance
 |              maxits : int
 |                  Maximum number of iterations
 |      
 |              See Also
 |              --------
 |              setTolerances, petsc.KSPGetTolerances
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:971
 |  
 |  getType(...)
 |      KSP.getType(self) -> str
 |      Return the KSP type as a string from the `KSP` object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:509
 |  
 |  getWorkVecs(...)
 |      KSP.getWorkVecs(self, right: int | None = None, left: int | None = None) -> tuple[list[Vec], list[Vec]] | list[Vec] | None
 |      Create working vectors.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              right
 |                  Number of right hand vectors to allocate.
 |              left
 |                  Number of left hand vectors to allocate.
 |      
 |              Returns
 |              -------
 |              R : list of Vec
 |                  List of correctly allocated right hand vectors.
 |              L : list of Vec
 |                  List of correctly allocated left hand vectors.
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1993
 |  
 |  logConvergenceHistory(...)
 |      KSP.logConvergenceHistory(self, rnorm: float) -> None
 |      Add residual to convergence history.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              rnorm
 |                  Residual norm to be added to convergence history.
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1203
 |  
 |  matSolve(...)
 |      KSP.matSolve(self, B: Mat, X: Mat) -> None
 |      Solve a linear system with multiple right-hand sides.
 |      
 |              Collective.
 |      
 |              These are stored as a `Mat.Type.DENSE`. Unlike `solve`,
 |              ``B`` and ``X`` must be different matrices.
 |      
 |              Parameters
 |              ----------
 |              B
 |                  Block of right-hand sides.
 |              X
 |                  Block of solutions.
 |      
 |              See Also
 |              --------
 |              solve, petsc.KSPMatSolve
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1808
 |  
 |  matSolveTranspose(...)
 |      KSP.matSolveTranspose(self, B: Mat, X: Mat) -> None
 |      Solve the transpose of a linear system with multiple RHS.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              B
 |                  Block of right-hand sides.
 |              X
 |                  Block of solutions.
 |      
 |              See Also
 |              --------
 |              solveTranspose, petsc.KSPMatSolve
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1830
 |  
 |  monitor(...)
 |      KSP.monitor(self, its: int, rnorm: float) -> None
 |      Run the user provided monitor routines, if they exist.
 |      
 |              Collective.
 |      
 |              Notes
 |              -----
 |              This routine is called by the `KSP` implementations. It does not
 |              typically need to be called by the user.
 |      
 |              See Also
 |              --------
 |              setMonitor, petsc.KSPMonitor
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1294
 |  
 |  monitorCancel(...)
 |      KSP.monitorCancel(self) -> None
 |      Clear all monitors for a `KSP` object.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, getMonitor, setMonitor, monitor, petsc.KSPMonitorCancel
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1279
 |  
 |  reset(...)
 |      KSP.reset(self) -> None
 |      Resets a KSP context.
 |      
 |              Collective.
 |      
 |              Resets a KSP context to the ``kspsetupcalled = 0`` state and
 |              removes any allocated Vecs and Mats.
 |      
 |              See Also
 |              --------
 |              petsc.KSPReset
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1607
 |  
 |  setAppCtx(...)
 |      KSP.setAppCtx(self, appctx: Any) -> None
 |      Set the optional user-defined context for the linear solver.
 |      
 |              Not collective.
 |      
 |              Parameters
 |              ----------
 |              appctx
 |                  The user defined context
 |      
 |              Notes
 |              -----
 |              The user context is a way for users to attach any information
 |              to the `KSP` that they may need later when interacting with
 |              the solver.
 |      
 |              See Also
 |              --------
 |              getAppCtx
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:617
 |  
 |  setComputeEigenvalues(...)
 |      KSP.setComputeEigenvalues(self, flag: bool) -> None
 |      Set a flag to compute eigenvalues.
 |      
 |              Logically collective.
 |      
 |              Set a flag so that the extreme eigenvalues values will be
 |              calculated via a Lanczos or Arnoldi process as the linear
 |              system is solved.
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  Boolean whether to compute eigenvalues (or not).
 |      
 |              Notes
 |              -----
 |              Currently this option is not valid for all iterative methods.
 |      
 |              See Also
 |              --------
 |              getComputeEigenvalues, petsc.KSPSetComputeEigenvalues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1404
 |  
 |  setComputeOperators(...)
 |      KSP.setComputeOperators(self, operators: KSPOperatorsFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set routine to compute the linear operators.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              operators
 |                  Function which computes the operators.
 |              args
 |                  Positional arguments for callback function ``operators``.
 |              kargs
 |                  Keyword arguments for callback function ``operators``.
 |      
 |              Notes
 |              -----
 |              The user provided function `operators` will be called
 |              automatically at the very next call to `solve`. It will NOT
 |              be called at future `solve` calls unless either
 |              `setComputeOperators` or `setOperators` is called
 |              before that `solve` is called. This allows the same system
 |              to be solved several times with different right-hand side
 |              functions, but is a confusing API since one might expect it to
 |              be called for each `solve`.
 |      
 |              To reuse the same preconditioner for the next `solve` and
 |              not compute a new one based on the most recently computed
 |              matrix call `petsc.KSPSetReusePreconditioner`.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, solve, setOperators, petsc.KSPSetComputeOperators
 |              petsc.KSPSetReusePreconditioner
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:764
 |  
 |  setComputeRHS(...)
 |      KSP.setComputeRHS(self, rhs: KSPRHSFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set routine to compute the right-hand side of the linear system.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              rhs
 |                  Function which computes the right-hand side.
 |              args
 |                  Positional arguments for callback function ``rhs``.
 |              kargs
 |                  Keyword arguments for callback function ``rhs``.
 |      
 |              Notes
 |              -----
 |              The routine you provide will be called each time you call `solve`
 |              to prepare the new right-hand side for that solve.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, solve, petsc.KSPSetComputeRHS
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:730
 |  
 |  setComputeSingularValues(...)
 |      KSP.setComputeSingularValues(self, flag: bool) -> None
 |      Set flag to calculate singular values.
 |      
 |              Logically collective.
 |      
 |              Set a flag so that the extreme singular values will be
 |              calculated via a Lanczos or Arnoldi process as the linear
 |              system is solved.
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  Boolean whether to compute singular values (or not).
 |      
 |              Notes
 |              -----
 |              Currently this option is not valid for all iterative methods.
 |      
 |              See Also
 |              --------
 |              getComputeSingularValues, petsc.KSPSetComputeSingularValues
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1448
 |  
 |  setConvergedReason(...)
 |      KSP.setConvergedReason(self, reason: KSP.ConvergedReason) -> None
 |      Use `reason` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1871
 |  
 |  setConvergenceHistory(...)
 |      KSP.setConvergenceHistory(self, length: int | None = None, reset: bool = False) -> None
 |      Set the array used to hold the residual history.
 |      
 |              Not collective.
 |      
 |              If set, this array will contain the residual norms computed at
 |              each iteration of the solver.
 |      
 |              Parameters
 |              ----------
 |              length
 |                  Length of array to store history in.
 |              reset
 |                  `True` indicates the history counter is reset to zero for
 |                  each new linear solve.
 |      
 |              Notes
 |              -----
 |              If ``length`` is not provided or `None` then a default array
 |              of length 10000 is allocated.
 |      
 |              If the array is not long enough then once the iterations is
 |              longer than the array length `solve` stops recording the
 |              history.
 |      
 |              See Also
 |              --------
 |              getConvergenceHistory, petsc.KSPSetResidualHistory
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1144
 |  
 |  setConvergenceTest(...)
 |      KSP.setConvergenceTest(self, converged: KSPConvergenceTestFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set the function to be used to determine convergence.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              converged
 |                  Callback which computes the convergence.
 |              args
 |                  Positional arguments for callback function.
 |              kargs
 |                  Keyword arguments for callback function.
 |      
 |              Notes
 |              -----
 |              Must be called after the KSP type has been set so put this
 |              after a call to `setType`, or `setFromOptions`.
 |      
 |              The default is a combination of relative and absolute
 |              tolerances. The residual value that is tested may be an
 |              approximation; routines that need exact values should compute
 |              them.
 |      
 |              See Also
 |              --------
 |              addConvergenceTest, ConvergedReason, setTolerances,
 |              getConvergenceTest, buildResidual,
 |              petsc.KSPSetConvergenceTest, petsc.KSPConvergedDefault
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1000
 |  
 |  setDM(...)
 |      KSP.setDM(self, dm: DM) -> None
 |      Set the `DM` that may be used by some preconditioners.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              dm
 |                  The `DM` object, cannot be `None`.
 |      
 |              Notes
 |              -----
 |              If this is used then the `KSP` will attempt to use the `DM` to
 |              create the matrix and use the routine set with
 |              `DM.setKSPComputeOperators`. Use ``setDMActive(False)``
 |              to instead use the matrix you have provided with
 |              `setOperators`.
 |      
 |              A `DM` can only be used for solving one problem at a time
 |              because information about the problem is stored on the `DM`,
 |              even when not using interfaces like
 |              `DM.setKSPComputeOperators`. Use `DM.clone` to get a distinct
 |              `DM` when solving different problems using the same function
 |              space.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, DM, DM.setKSPComputeOperators, setOperators, DM.clone
 |              petsc.KSPSetDM
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:671
 |  
 |  setDMActive(...)
 |      KSP.setDMActive(self, flag: bool) -> None
 |      `DM` should be used to generate system matrix & RHS vector.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  Whether to use the `DM`.
 |      
 |              Notes
 |              -----
 |              By default `setDM` sets the `DM` as active, call
 |              ``setDMActive(False)`` after ``setDM(dm)`` to not
 |              have the `KSP` object use the `DM` to generate the matrices.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, DM, setDM, petsc.KSPSetDMActive
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:704
 |  
 |  setErrorIfNotConverged(...)
 |      KSP.setErrorIfNotConverged(self, flag: bool) -> None
 |      Cause `solve` to generate an error if not converged.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  `True` enables this behavior.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetErrorIfNotConverged
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1928
 |  
 |  setFromOptions(...)
 |      KSP.setFromOptions(self) -> None
 |      Set `KSP` options from the options database.
 |      
 |              Collective.
 |      
 |              This routine must be called before `setUp` if the user is
 |              to be allowed to set the Krylov type.
 |      
 |              See Also
 |              --------
 |              petsc_options, petsc.KSPSetFromOptions
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:600
 |  
 |  setGMRESRestart(...)
 |      KSP.setGMRESRestart(self, restart: int) -> None
 |      Set number of iterations at which KSP restarts.
 |      
 |              Logically collective.
 |      
 |              Suitable KSPs are: KSPGMRES, KSPFGMRES and KSPLGMRES.
 |      
 |              Parameters
 |              ----------
 |              restart
 |                  Integer restart value.
 |      
 |              See Also
 |              --------
 |              petsc.KSPGMRESSetRestart
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2132
 |  
 |  setHPDDMType(...)
 |      KSP.setHPDDMType(self, hpddm_type: HPDDMType) -> None
 |      Set the Krylov solver type.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              hpddm_type
 |                  The type of Krylov solver to use.
 |      
 |              See Also
 |              --------
 |              petsc.KSPHPDDMSetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1896
 |  
 |  setInitialGuessKnoll(...)
 |      KSP.setInitialGuessKnoll(self, flag: bool) -> None
 |      Tell solver to use `PC.apply` to compute the initial guess.
 |      
 |              Logically collective.
 |      
 |              This is the Knoll trick.
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  `True` uses Knoll trick.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetInitialGuessKnoll
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1530
 |  
 |  setInitialGuessNonzero(...)
 |      KSP.setInitialGuessNonzero(self, flag: bool) -> None
 |      Tell the iterative solver that the initial guess is nonzero.
 |      
 |              Logically collective.
 |      
 |              Otherwise KSP assumes the initial guess is to be zero (and thus
 |              zeros it out before solving).
 |      
 |              Parameters
 |              ----------
 |              flag
 |                  `True` indicates the guess is non-zero, `False`
 |                  indicates the guess is zero.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetInitialGuessNonzero
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1494
 |  
 |  setIterationNumber(...)
 |      KSP.setIterationNumber(self, its: int) -> None
 |      Use `its` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1849
 |  
 |  setMonitor(...)
 |      KSP.setMonitor(self, monitor: KSPMonitorFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set additional function to monitor the residual.
 |      
 |              Logically collective.
 |      
 |              Set an ADDITIONAL function to be called at every iteration to
 |              monitor the residual/error etc.
 |      
 |              Parameters
 |              ----------
 |              monitor
 |                  Callback which monitors the convergence.
 |              args
 |                  Positional arguments for callback function.
 |              kargs
 |                  Keyword arguments for callback function.
 |      
 |              Notes
 |              -----
 |              The default is to do nothing. To print the residual, or
 |              preconditioned residual if
 |              ``setNormType(NORM_PRECONDITIONED)`` was called, use
 |              `monitor` as the monitoring routine, with a
 |              `PETSc.Viewer.ASCII` as the context.
 |      
 |              Several different monitoring routines may be set by calling
 |              `setMonitor` multiple times; all will be called in the order
 |              in which they were set.
 |      
 |              See Also
 |              --------
 |              petsc_options, getMonitor, monitor, monitorCancel, petsc.KSPMonitorSet
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1219
 |  
 |  setNormType(...)
 |      KSP.setNormType(self, normtype: NormType) -> None
 |      Set the norm that is used for convergence testing.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              normtype
 |                  The norm type to use (see `NormType`).
 |      
 |              Notes
 |              -----
 |              Not all combinations of preconditioner side (see
 |              `setPCSide`) and norm type are supported by all Krylov
 |              methods. If only one is set, PETSc tries to automatically
 |              change the other to find a compatible pair. If no such
 |              combination is supported, PETSc will generate an error.
 |      
 |              See Also
 |              --------
 |              NormType, petsc_options, setUp, solve, destroy, setPCSide, getPCSide
 |              NormType, petsc.KSPSetNormType, petsc.KSPConvergedSkip
 |              petsc.KSPSetCheckNormIteration
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1363
 |  
 |  setOperators(...)
 |      KSP.setOperators(self, A: Mat | None = None, P: Mat | None = None) -> None
 |      Set matrix associated with the linear system.
 |      
 |              Collective.
 |      
 |              Set the matrix associated with the linear system and a
 |              (possibly) different one from which the preconditioner will be
 |              built.
 |      
 |              Parameters
 |              ----------
 |              A
 |                  Matrix that defines the linear system.
 |              P
 |                  Matrix to be used in constructing the preconditioner,
 |                  usually the same as ``A``.
 |      
 |              Notes
 |              -----
 |              If you know the operator ``A`` has a null space you can use
 |              `Mat.setNullSpace` and `Mat.setTransposeNullSpace` to supply the
 |              null space to ``A`` and the `KSP` solvers will automatically use
 |              that null space as needed during the solution process.
 |      
 |              All future calls to `setOperators` must use the same size
 |              matrices!
 |      
 |              Passing `None` for ``A`` or ``P`` removes the matrix that is
 |              currently used.
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, solve, setComputeOperators, petsc.KSPSetOperators
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:809
 |  
 |  setOptionsPrefix(...)
 |      KSP.setOptionsPrefix(self, prefix: str | None) -> None
 |      Set the prefix used for all `KSP` options in the database.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              prefix
 |                  The options prefix.
 |      
 |              Notes
 |              -----
 |              A hyphen (-) must NOT be given at the beginning of the prefix
 |              name. The first character of all runtime options is
 |              AUTOMATICALLY the hyphen. For example, to distinguish between
 |              the runtime options for two different `KSP` contexts, one could
 |              call
 |              ```
 |              KSPSetOptionsPrefix(ksp1, "sys1_")
 |              KSPSetOptionsPrefix(ksp2, "sys2_")
 |              ```
 |      
 |              This would enable use of different options for each system,
 |              such as
 |              ```
 |              -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
 |              -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
 |              ```
 |      
 |              See Also
 |              --------
 |              petsc_options, petsc.KSPSetOptionsPrefix
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:523
 |  
 |  setPC(...)
 |      KSP.setPC(self, pc: PC) -> None
 |      Set the preconditioner.
 |      
 |              Collective.
 |      
 |              Set the preconditioner to be used to calculate the application
 |              of the preconditioner on a vector.
 |      
 |              Parameters
 |              ----------
 |              pc
 |                  The preconditioner object
 |      
 |              See Also
 |              --------
 |              PETSc.KSP, getPC, petsc.KSPSetPC
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:877
 |  
 |  setPCSide(...)
 |      KSP.setPCSide(self, side: PC.Side) -> None
 |      Set the preconditioning side.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              side
 |                  The preconditioning side (see `PC.Side`).
 |      
 |              Notes
 |              -----
 |              Left preconditioning is used by default for most Krylov methods
 |              except `Type.FGMRES` which only supports right preconditioning.
 |      
 |              For methods changing the side of the preconditioner changes the
 |              norm type that is used, see `setNormType`.
 |      
 |              Symmetric preconditioning is currently available only for the
 |              `Type.QCG` method. Note, however, that symmetric preconditioning
 |              can be emulated by using either right or left preconditioning
 |              and a pre or post processing step.
 |      
 |              Setting the PC side often affects the default norm type. See
 |              `setNormType` for details.
 |      
 |              See Also
 |              --------
 |              PC.Side, petsc_options, getPCSide, setNormType, getNormType
 |              petsc.KSPSetPCSide
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1315
 |  
 |  setPostSolve(...)
 |      KSP.setPostSolve(self, postsolve: KSPPostSolveFunction | None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set the function that is called at the end of each `KSP.solve`.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              postsolve
 |                  The callback function.
 |              args
 |                  Positional arguments for the callback function.
 |              kargs
 |                  Keyword arguments for the callback function.
 |      
 |              See Also
 |              --------
 |              solve, petsc.KSPSetPreSolve, petsc.KSPSetPostSolve
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1670
 |  
 |  setPreSolve(...)
 |      KSP.setPreSolve(self, presolve: KSPPreSolveFunction | None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None
 |      Set the function that is called at the beginning of each `KSP.solve`.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              presolve
 |                  The callback function.
 |              args
 |                  Positional arguments for the callback function.
 |              kargs
 |                  Keyword arguments for the callback function.
 |      
 |              See Also
 |              --------
 |              solve, petsc.KSPSetPreSolve, petsc.KSPSetPostSolve
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1637
 |  
 |  setPythonContext(...)
 |      KSP.setPythonContext(self, context: Any | None = None) -> None
 |      Set the instance of the class implementing Python methods.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc_python_ksp, getPythonContext
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2183
 |  
 |  setPythonType(...)
 |      KSP.setPythonType(self, py_type: str) -> None
 |      Set the fully qualified Python name of the class to be used.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc_python_ksp, setPythonContext, getPythonType
 |              petsc.KSPPythonSetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:2210
 |  
 |  setResidualNorm(...)
 |      KSP.setResidualNorm(self, rnorm: float) -> None
 |      Use `norm` property.
 |      Source code at petsc4py/PETSc/KSP.pyx:1860
 |  
 |  setTolerances(...)
 |      KSP.setTolerances(self, rtol: float | None = None, atol: float | None = None, divtol: float | None = None, max_it: int | None = None) -> None
 |      Set various tolerances used by the KSP convergence testers.
 |      
 |              Logically collective.
 |      
 |              Set the relative, absolute, divergence, and maximum iteration
 |              tolerances used by the default KSP convergence testers.
 |      
 |              Parameters
 |              ----------
 |              rtol
 |                  The relative convergence tolerance, relative decrease in
 |                  the (possibly preconditioned) residual norm.
 |                  Or `DETERMINE` to use the value when
 |                  the object's type was set.
 |              atol
 |                  The absolute convergence tolerance absolute size of the
 |                  (possibly preconditioned) residual norm.
 |                  Or `DETERMINE` to use the value when
 |                  the object's type was set.
 |              dtol
 |                  The divergence tolerance, amount (possibly preconditioned)
 |                  residual norm can increase before
 |                  `petsc.KSPConvergedDefault` concludes that the method is
 |                  diverging.
 |                  Or `DETERMINE` to use the value when
 |                  the object's type was set.
 |              max_it
 |                  Maximum number of iterations to use.
 |                  Or `DETERMINE` to use the value when
 |                  the object's type was set.
 |      
 |              Notes
 |              -----
 |              Use `None` to retain the default value of any of the
 |              tolerances.
 |      
 |              See Also
 |              --------
 |              petsc_options, getTolerances, setConvergenceTest
 |              petsc.KSPSetTolerances, petsc.KSPConvergedDefault
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:914
 |  
 |  setType(...)
 |      KSP.setType(self, ksp_type: Type | str) -> None
 |      Build the `KSP` data structure for a particular `Type`.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              ksp_type
 |                  KSP Type object
 |      
 |              Notes
 |              -----
 |              See `Type` for available methods (for instance, `Type.CG` or
 |              `Type.GMRES`).
 |      
 |              Normally, it is best to use the `setFromOptions` command
 |              and then set the KSP type from the options database rather than
 |              by using this routine. Using the options database provides the
 |              user with maximum flexibility in evaluating the many different
 |              Krylov methods. This method is provided for those situations
 |              where it is necessary to set the iterative solver independently
 |              of the command line or options database. This might be the
 |              case, for example, when the choice of iterative solver changes
 |              during the execution of the program, and the user's application
 |              is taking responsibility for choosing the appropriate method.
 |              In other words, this routine is not for beginners.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetType
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:473
 |  
 |  setUp(...)
 |      KSP.setUp(self) -> None
 |      Set up internal data structures for an iterative solver.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetUp
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1595
 |  
 |  setUpOnBlocks(...)
 |      KSP.setUpOnBlocks(self) -> None
 |      Set up the preconditioner for each block in a block method.
 |      
 |              Collective.
 |      
 |              Methods include: block Jacobi, block Gauss-Seidel, and
 |              overlapping Schwarz methods.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetUpOnBlocks
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1622
 |  
 |  setUseFischerGuess(...)
 |      KSP.setUseFischerGuess(self, model: int, size: int) -> None
 |      Use the Paul Fischer algorithm to compute initial guesses.
 |      
 |              Logically collective.
 |      
 |              Use the Paul Fischer algorithm or its variants to compute
 |              initial guesses for a set of solves with related right hand
 |              sides.
 |      
 |              Parameters
 |              ----------
 |              model
 |                  Use model ``1``, model ``2``, model ``3``, any other number
 |                  to turn it off.
 |              size
 |                  Size of subspace used to generate initial guess.
 |      
 |              See Also
 |              --------
 |              petsc.KSPSetUseFischerGuess
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1567
 |  
 |  solve(...)
 |      KSP.solve(self, b: Vec, x: Vec) -> None
 |      Solve the linear system.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              b
 |                  Right hand side vector.
 |              x
 |                  Solution vector.
 |      
 |              Notes
 |              -----
 |              If one uses `setDM` then ``x`` or ``b`` need not be passed. Use
 |              `getSolution` to access the solution in this case.
 |      
 |              The operator is specified with `setOperators`.
 |      
 |              `solve` will normally return without generating an error
 |              regardless of whether the linear system was solved or if
 |              constructing the preconditioner failed. Call
 |              `getConvergedReason` to determine if the solver converged or
 |              failed and why. The option ``-ksp_error_if_not_converged`` or
 |              function `setErrorIfNotConverged` will cause `solve` to error
 |              as soon as an error occurs in the linear solver. In inner
 |              solves, ``DIVERGED_MAX_IT`` is not treated as an error
 |              because when using nested solvers it may be fine that inner
 |              solvers in the preconditioner do not converge during the
 |              solution process.
 |      
 |              The number of iterations can be obtained from `its`.
 |      
 |              If you provide a matrix that has a `Mat.setNullSpace` and
 |              `Mat.setTransposeNullSpace` this will use that information to
 |              solve singular systems in the least squares sense with a norm
 |              minimizing solution.
 |      
 |              Ax = b where b = bₚ + bₜ where bₜ is not in the range of A
 |              (and hence by the fundamental theorem of linear algebra is in
 |              the nullspace(Aᵀ), see `Mat.setNullSpace`.
 |      
 |              KSP first removes bₜ producing the linear system Ax = bₚ (which
 |              has multiple solutions) and solves this to find the ∥x∥
 |              minimizing solution (and hence it finds the solution x
 |              orthogonal to the nullspace(A). The algorithm is simply in each
 |              iteration of the Krylov method we remove the nullspace(A) from
 |              the search direction thus the solution which is a linear
 |              combination of the search directions has no component in the
 |              nullspace(A).
 |      
 |              We recommend always using `Type.GMRES` for such singular
 |              systems. If nullspace(A) = nullspace(Aᵀ) (note symmetric
 |              matrices always satisfy this property) then both left and right
 |              preconditioning will work If nullspace(A) != nullspace(Aᵀ) then
 |              left preconditioning will work but right preconditioning may
 |              not work (or it may).
 |      
 |              If using a direct method (e.g., via the KSP solver
 |              `Type.PREONLY` and a preconditioner such as `PC.Type.LU` or
 |              `PC.Type.ILU`, then its=1. See `setTolerances` for more details.
 |      
 |              **Understanding Convergence**
 |      
 |              The routines `setMonitor` and `computeEigenvalues` provide
 |              information on additional options to monitor convergence and
 |              print eigenvalue information.
 |      
 |              See Also
 |              --------
 |              create, setUp, destroy, setTolerances, is_converged, solveTranspose, its
 |              Mat.setNullSpace, Mat.setTransposeNullSpace, Type,
 |              setErrorIfNotConverged petsc_options, petsc.KSPSolve
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1703
 |  
 |  solveTranspose(...)
 |      KSP.solveTranspose(self, b: Vec, x: Vec) -> None
 |      Solve the transpose of a linear system.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              b
 |                  Right hand side vector.
 |              x
 |                  Solution vector.
 |      
 |              Notes
 |              -----
 |              For complex numbers this solve the non-Hermitian transpose
 |              system.
 |      
 |              See Also
 |              --------
 |              solve, petsc.KSPSolveTranspose
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:1784
 |  
 |  view(...)
 |      KSP.view(self, viewer: Viewer | None = None) -> None
 |      Print the KSP data structure.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              viewer
 |                  Viewer used to display the KSP.
 |      
 |              See Also
 |              --------
 |              petsc.KSPView
 |      
 |              
 |      Source code at petsc4py/PETSc/KSP.pyx:425
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  __new__(*args, **kwargs) from builtins.type
 |      Create and return a new object.  See help(type) for accurate signature.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  appctx
 |      KSP.appctx: Any
 |      The solver application context.
 |      Source code at petsc4py/PETSc/KSP.pyx:2242
 |  
 |  atol
 |      KSP.atol: float
 |      The absolute tolerance of the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2335
 |  
 |  divtol
 |      KSP.divtol: float
 |      The divergence tolerance of the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2343
 |  
 |  dm
 |      KSP.dm: DM
 |      The solver `DM`.
 |      Source code at petsc4py/PETSc/KSP.pyx:2252
 |  
 |  guess_knoll
 |      KSP.guess_knoll: bool
 |      Whether solver uses Knoll trick.
 |      Source code at petsc4py/PETSc/KSP.pyx:2294
 |  
 |  guess_nonzero
 |      KSP.guess_nonzero: bool
 |      Whether guess is non-zero.
 |      Source code at petsc4py/PETSc/KSP.pyx:2286
 |  
 |  history
 |      KSP.history: ndarray
 |      The convergence history of the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2377
 |  
 |  is_converged
 |      KSP.is_converged: bool
 |      Boolean indicating if the solver has converged.
 |      Source code at petsc4py/PETSc/KSP.pyx:2397
 |  
 |  is_diverged
 |      KSP.is_diverged: bool
 |      Boolean indicating if the solver has failed.
 |      Source code at petsc4py/PETSc/KSP.pyx:2402
 |  
 |  is_iterating
 |      KSP.is_iterating: bool
 |      Boolean indicating if the solver has not converged yet.
 |      Source code at petsc4py/PETSc/KSP.pyx:2392
 |  
 |  its
 |      KSP.its: int
 |      The current number of iterations the solver has taken.
 |      Source code at petsc4py/PETSc/KSP.pyx:2361
 |  
 |  mat_op
 |      KSP.mat_op: Mat
 |      The system matrix operator.
 |      Source code at petsc4py/PETSc/KSP.pyx:2274
 |  
 |  mat_pc
 |      KSP.mat_pc: Mat
 |      The preconditioner operator.
 |      Source code at petsc4py/PETSc/KSP.pyx:2279
 |  
 |  max_it
 |      KSP.max_it: int
 |      The maximum number of iteration the solver may take.
 |      Source code at petsc4py/PETSc/KSP.pyx:2351
 |  
 |  norm
 |      KSP.norm: float
 |      The norm of the residual at the current iteration.
 |      Source code at petsc4py/PETSc/KSP.pyx:2369
 |  
 |  norm_type
 |      KSP.norm_type: NormType
 |      The norm used by the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2317
 |  
 |  pc
 |      KSP.pc: PC
 |      The `PC` of the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2304
 |  
 |  pc_side
 |      KSP.pc_side: PC.Side
 |      The side on which preconditioning is performed.
 |      Source code at petsc4py/PETSc/KSP.pyx:2309
 |  
 |  reason
 |      KSP.reason: KSP.ConvergedReason
 |      The converged reason.
 |      Source code at petsc4py/PETSc/KSP.pyx:2384
 |  
 |  rtol
 |      KSP.rtol: float
 |      The relative tolerance of the solver.
 |      Source code at petsc4py/PETSc/KSP.pyx:2327
 |  
 |  vec_rhs
 |      KSP.vec_rhs: Vec
 |      The right-hand side vector.
 |      Source code at petsc4py/PETSc/KSP.pyx:2267
 |  
 |  vec_sol
 |      KSP.vec_sol: Vec
 |      The solution vector.
 |      Source code at petsc4py/PETSc/KSP.pyx:2262
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  ConvergedReason = <class 'petsc4py.PETSc.KSPConvergedReason'>
 |      KSP Converged Reason.
 |      
 |      `CONVERGED_ITERATING`
 |          Still iterating
 |      `ITERATING`
 |          Still iterating
 |      
 |      `CONVERGED_RTOL_NORMAL`
 |          Undocumented.
 |      `CONVERGED_ATOL_NORMAL`
 |          Undocumented.
 |      `CONVERGED_RTOL`
 |          ∥r∥ <= rtolnorm(b) or rtolnorm(b - Ax₀)
 |      `CONVERGED_ATOL`
 |          ∥r∥ <= atol
 |      `CONVERGED_ITS`
 |          Used by the `Type.PREONLY` solver after the single iteration of the
 |          preconditioner is applied. Also used when the
 |          `petsc.KSPConvergedSkip` convergence test routine is set in KSP.
 |      `CONVERGED_NEG_CURVE`
 |          Undocumented.
 |      `CONVERGED_STEP_LENGTH`
 |          Undocumented.
 |      `CONVERGED_HAPPY_BREAKDOWN`
 |          Undocumented.
 |      
 |      `DIVERGED_NULL`
 |          Undocumented.
 |      `DIVERGED_MAX_IT`
 |          Ran out of iterations before any convergence criteria was
 |          reached.
 |      `DIVERGED_DTOL`
 |          norm(r) >= dtol*norm(b)
 |      `DIVERGED_BREAKDOWN`
 |          A breakdown in the Krylov method was detected so the method
 |          could not continue to enlarge the Krylov space. Could be due to
 |          a singular matrix or preconditioner. In KSPHPDDM, this is also
 |          returned when some search directions within a block are
 |          colinear.
 |      `DIVERGED_BREAKDOWN_BICG`
 |          A breakdown in the KSPBICG method was detected so the method
 |          could not continue to enlarge the Krylov space.
 |      `DIVERGED_NONSYMMETRIC`
 |          It appears the operator or preconditioner is not symmetric and
 |          this Krylov method (`Type.CG`, `Type.MINRES`, `Type.CR`)
 |          requires symmetry.
 |      `DIVERGED_INDEFINITE_PC`
 |          It appears the preconditioner is indefinite (has both positive
 |          and negative eigenvalues) and this Krylov method (`Type.CG`)
 |          requires it to be positive definite.
 |      `DIVERGED_NANORINF`
 |          Undocumented.
 |      `DIVERGED_INDEFINITE_MAT`
 |          Undocumented.
 |      `DIVERGED_PCSETUP_FAILED`
 |          It was not possible to build or use the requested
 |          preconditioner. This is usually due to a zero pivot in a
 |          factorization. It can also result from a failure in a
 |          subpreconditioner inside a nested preconditioner such as
 |          `PC.Type.FIELDSPLIT`.
 |      
 |      See Also
 |      --------
 |      `petsc.KSPConvergedReason`
 |  
 |  
 |  HPDDMType = <class 'petsc4py.PETSc.KSPHPDDMType'>
 |      The *HPDDM* Krylov solver type.
 |  
 |  
 |  NormType = <class 'petsc4py.PETSc.KSPNormType'>
 |      KSP norm type.
 |      
 |      The available norm types are:
 |      
 |      `NONE`
 |          Skips computing the norm, this should generally only be used if
 |          you are using the Krylov method as a smoother with a fixed
 |          small number of iterations. Implicitly sets
 |          `petsc.KSPConvergedSkip` as KSP convergence test. Note that
 |          certain algorithms such as `Type.GMRES` ALWAYS require the norm
 |          calculation, for these methods the norms are still computed,
 |          they are just not used in the convergence test.
 |      `PRECONDITIONED`
 |          The default for left preconditioned solves, uses the l₂ norm of
 |          the preconditioned residual P⁻¹(b - Ax).
 |      `UNPRECONDITIONED`
 |          Uses the l₂ norm of the true b - Ax residual.
 |      `NATURAL`
 |          Supported by `Type.CG`, `Type.CR`, `Type.CGNE`, `Type.CGS`.
 |  
 |  
 |  Type = <class 'petsc4py.PETSc.KSPType'>
 |      KSP Type.
 |      
 |      The available types are:
 |      
 |      `RICHARDSON`
 |          The preconditioned Richardson iterative method
 |          `petsc.KSPRICHARDSON`.
 |      `CHEBYSHEV`
 |          The preconditioned Chebyshev iterative method.
 |          `petsc.KSPCHEBYSHEV`.
 |      `CG`
 |          The Preconditioned Conjugate Gradient (PCG) iterative method.
 |          `petsc.KSPCG`
 |      `GROPPCG`
 |          A pipelined conjugate gradient method (Gropp).
 |          `petsc.KSPGROPPCG`
 |      `PIPECG`
 |          A pipelined conjugate gradient method.
 |          `petsc.KSPPIPECG`
 |      `PIPECGRR`
 |          Pipelined Conjugate Gradients with Residual Replacement.
 |          `petsc.KSPPIPECGRR`
 |      `PIPELCG`
 |          Deep pipelined (length l) Conjugate Gradient method.
 |          `petsc.KSPPIPELCG`
 |      `PIPEPRCG`
 |          Pipelined predict-and-recompute conjugate gradient method.
 |          `petsc.KSPPIPEPRCG`
 |      `PIPECG2`
 |          Pipelined conjugate gradient method with a single non-blocking
 |          reduction per two iterations. `petsc.KSPPIPECG2`
 |      `CGNE`
 |          Applies the preconditioned conjugate gradient method to the
 |          normal equations without explicitly forming AᵀA. `petsc.KSPCGNE`
 |      `NASH`
 |          Conjugate gradient method subject to a constraint
 |          on the solution norm. `petsc.KSPNASH`
 |      `STCG`
 |          Conjugate gradient method subject to a constraint on the
 |          solution norm. `petsc.KSPSTCG`
 |      `GLTR`
 |          Conjugate gradient method subject to a constraint on the
 |          solution norm. `petsc.KSPGLTR`
 |      `FCG`
 |          Flexible Conjugate Gradient method (FCG). Unlike most KSP
 |          methods this allows the preconditioner to be nonlinear.
 |          `petsc.KSPFCG`
 |      `PIPEFCG`
 |          Pipelined, Flexible Conjugate Gradient method.
 |          `petsc.KSPPIPEFCG`
 |      `GMRES`
 |          Generalized Minimal Residual method with restart.
 |          `petsc.KSPGMRES`
 |      `PIPEFGMRES`
 |          Pipelined (1-stage) Flexible Generalized Minimal Residual
 |          method. `petsc.KSPPIPEFGMRES`
 |      `FGMRES`
 |          Implements the Flexible Generalized Minimal Residual method.
 |          `petsc.KSPFGMRES`
 |      `LGMRES`
 |          Augments the standard Generalized Minimal Residual method
 |          approximation space with approximations to the error from
 |          previous restart cycles. `petsc.KSPLGMRES`
 |      `DGMRES`
 |          Deflated Generalized Minimal Residual method. In this
 |          implementation, the adaptive strategy allows to switch to the
 |          deflated GMRES when the stagnation occurs. `petsc.KSPDGMRES`
 |      `PGMRES`
 |          Pipelined Generalized Minimal Residual method.
 |          `petsc.KSPPGMRES`
 |      `TCQMR`
 |          A variant of Quasi Minimal Residual (QMR).
 |          `petsc.KSPTCQMR`
 |      `BCGS`
 |          Stabilized version of Biconjugate Gradient (BiCGStab) method.
 |          `petsc.KSPBCGS`
 |      `IBCGS`
 |          Improved Stabilized version of BiConjugate Gradient (IBiCGStab)
 |          method in an alternative form to have only a single global
 |          reduction operation instead of the usual 3 (or 4).
 |          `petsc.KSPIBCGS`
 |      `QMRCGS`
 |          Quasi- Minimal Residual variant of the Bi-CGStab algorithm
 |          (QMRCGStab) method. `petsc.KSPQMRCGS`
 |      `FBCGS`
 |          Flexible Stabilized version of BiConjugate Gradient (BiCGStab)
 |          method. `petsc.KSPFBCGS`
 |      `FBCGSR`
 |          A mathematically equivalent variant of flexible stabilized
 |          BiConjugate Gradient (BiCGStab). `petsc.KSPFBCGSR`
 |      `BCGSL`
 |          Variant of the L-step stabilized BiConjugate Gradient
 |          (BiCGStab(L)) algorithm. Uses "L-step" Minimal Residual (MR)
 |          polynomials. The variation concerns cases when some parameters
 |          are negative due to round-off. `petsc.KSPBCGSL`
 |      `PIPEBCGS`
 |          Pipelined stabilized BiConjugate Gradient (BiCGStab) method.
 |          `petsc.KSPPIPEBCGS`
 |      `CGS`
 |          Conjugate Gradient Squared method.
 |          `petsc.KSPCGS`
 |      `TFQMR`
 |          A Transpose Tree Quasi- Minimal Residual (QMR).
 |          `petsc.KSPCR`
 |      `CR`
 |          (Preconditioned) Conjugate Residuals (CR) method.
 |          `petsc.KSPCR`
 |      `PIPECR`
 |          Pipelined Conjugate Residual (CR) method.
 |          `petsc.KSPPIPECR`
 |      `LSQR`
 |          Least squares solver.
 |          `petsc.KSPLSQR`
 |      `PREONLY`
 |          Applies ONLY the preconditioner exactly once. This may be used
 |          in inner iterations, where it is desired to allow multiple
 |          iterations as well as the "0-iteration" case. It is commonly
 |          used with the direct solver preconditioners like PCLU and
 |          PCCHOLESKY. There is an alias of KSPNONE.
 |          `petsc.KSPPREONLY`
 |      `NONE`
 |          No solver
 |          ``KSPNONE``
 |      `QCG`
 |          Conjugate Gradient (CG) method subject to a constraint on the
 |          solution norm. `petsc.KSPQCG`
 |      `BICG`
 |          Implements the Biconjugate gradient method (BiCG).
 |          Similar to running the conjugate gradient on the normal equations.
 |          `petsc.KSPBICG`
 |      `MINRES`
 |          Minimum Residual (MINRES) method.
 |          `petsc.KSPMINRES`
 |      `SYMMLQ`
 |          Symmetric LQ method (SymmLQ). Uses LQ decomposition (lower
 |          trapezoidal).
 |          `petsc.KSPSYMMLQ`
 |      `LCD`
 |          Left Conjugate Direction (LCD) method.
 |          `petsc.KSPLCD`
 |      `PYTHON`
 |          Python shell solver. Call Python function to implement solver.
 |          ``KSPPYTHON``
 |      `GCR`
 |          Preconditioned flexible Generalized Conjugate Residual (GCR)
 |          method.
 |          `petsc.KSPGCR`
 |      `PIPEGCR`
 |          Pipelined Generalized Conjugate Residual method.
 |          `petsc.KSPPIPEGCR`
 |      `TSIRM`
 |          Two-Stage Iteration with least-squares Residual Minimization
 |          method. `petsc.KSPTSIRM`
 |      `CGLS`
 |          Conjugate Gradient method for Least-Squares problems. Supports
 |          non-square (rectangular) matrices. `petsc.KSPCGLS`
 |      `FETIDP`
 |          Dual-Primal (DP) Finite Element Tearing and Interconnect (FETI)
 |          method. `petsc.KSPFETIDP`
 |      `HPDDM`
 |          Interface with the HPDDM library. This KSP may be used to
 |          further select methods that are currently not implemented
 |          natively in PETSc, e.g., GCRODR, a recycled Krylov
 |          method which is similar to KSPLGMRES. `petsc.KSPHPDDM`
 |      
 |      See Also
 |      --------
 |      petsc_options, petsc.KSPType
 |  
 |  
 |  __pyx_vtable__ = <capsule object NULL>
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from Object:
 |  
 |  __bool__(self, /)
 |      True if self else False
 |  
 |  __copy__(...)
 |      Object.__copy__(self)
 |      Source code at petsc4py/PETSc/Object.pyx:32
 |  
 |  __deepcopy__(...)
 |      Object.__deepcopy__(self, memo: dict)
 |      Source code at petsc4py/PETSc/Object.pyx:40
 |  
 |  __eq__(self, value, /)
 |      Return self==value.
 |  
 |  __ge__(self, value, /)
 |      Return self>=value.
 |  
 |  __gt__(self, value, /)
 |      Return self>value.
 |  
 |  __le__(self, value, /)
 |      Return self<=value.
 |  
 |  __lt__(self, value, /)
 |      Return self<value.
 |  
 |  __ne__(self, value, /)
 |      Return self!=value.
 |  
 |  compose(...)
 |      Object.compose(self, name: str | None, obj: Object) -> None
 |      Associate a PETSc object using a key string.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              name
 |                  The string identifying the object to be composed.
 |              obj
 |                  The object to be composed.
 |      
 |              See Also
 |              --------
 |              query, petsc.PetscObjectCompose
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:324
 |  
 |  decRef(...)
 |      Object.decRef(self) -> int
 |      Decrement the object reference count.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              getRefCount, petsc.PetscObjectDereference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:384
 |  
 |  destroyOptionsHandlers(...)
 |      Object.destroyOptionsHandlers(self) -> None
 |      Clear all the option handlers.
 |      
 |              Collective.
 |      
 |              See Also
 |              --------
 |              petsc_options, setOptionsHandler, petsc.PetscObjectDestroyOptionsHandlers
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:208
 |  
 |  getAttr(...)
 |      Object.getAttr(self, name: str) -> object
 |      Return the attribute associated with a given name.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setAttr, getDict
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:403
 |  
 |  getClassId(...)
 |      Object.getClassId(self) -> int
 |      Return the class identifier of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetClassId
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:265
 |  
 |  getClassName(...)
 |      Object.getClassName(self) -> str
 |      Return the class name of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetClassName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:279
 |  
 |  getComm(...)
 |      Object.getComm(self) -> Comm
 |      Return the communicator of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetComm
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:223
 |  
 |  getDict(...)
 |      Object.getDict(self) -> dict
 |      Return the dictionary of attributes.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setAttr, getAttr
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:431
 |  
 |  getId(...)
 |      Object.getId(self) -> int
 |      Return the unique identifier of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetId
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:308
 |  
 |  getName(...)
 |      Object.getName(self) -> str
 |      Return the name of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:237
 |  
 |  getRefCount(...)
 |      Object.getRefCount(self) -> int
 |      Return the reference count of the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectGetReference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:293
 |  
 |  getTabLevel(...)
 |      Object.getTabLevel(self) -> None
 |      Return the PETSc object tab level.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              setTabLevel, incrementTabLevel, petsc.PetscObjectGetTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:513
 |  
 |  incRef(...)
 |      Object.incRef(self) -> int
 |      Increment the object reference count.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              getRefCount, petsc.PetscObjectReference
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:367
 |  
 |  incrementTabLevel(...)
 |      Object.incrementTabLevel(self, tab: int, parent: Object | None = None) -> None
 |      Increment the PETSc object tab level.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              setTabLevel, getTabLevel, petsc.PetscObjectIncrementTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:486
 |  
 |  query(...)
 |      Object.query(self, name: str) -> Object
 |      Query for the PETSc object associated with a key string.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              compose, petsc.PetscObjectQuery
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:347
 |  
 |  setAttr(...)
 |      Object.setAttr(self, name: str, attr: object) -> None
 |      Set an the attribute associated with a given name.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              getAttr, getDict
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:417
 |  
 |  setName(...)
 |      Object.setName(self, name: str | None) -> None
 |      Associate a name to the object.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              petsc.PetscObjectSetName
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:251
 |  
 |  setOptionsHandler(...)
 |      Object.setOptionsHandler(self, handler: PetscOptionsHandlerFunction | None) -> None
 |      Set the callback for processing extra options.
 |      
 |              Logically collective.
 |      
 |              Parameters
 |              ----------
 |              handler
 |                  The callback function, called at the end of `setFromOptions`.
 |      
 |              See Also
 |              --------
 |              petsc_options, setFromOptions, petsc.PetscObjectAddOptionsHandler
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:187
 |  
 |  setTabLevel(...)
 |      Object.setTabLevel(self, level: int) -> None
 |      Set the PETSc object tab level.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              incrementTabLevel, getTabLevel, petsc.PetscObjectSetTabLevel
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:500
 |  
 |  stateGet(...)
 |      Object.stateGet(self) -> int
 |      Return the PETSc object state.
 |      
 |              Not collective.
 |      
 |              See Also
 |              --------
 |              stateSet, stateIncrease, petsc.PetscObjectStateGet
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:457
 |  
 |  stateIncrease(...)
 |      Object.stateIncrease(self) -> None
 |      Increment the PETSc object state.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              stateGet, stateSet, petsc.PetscObjectStateIncrease
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:445
 |  
 |  stateSet(...)
 |      Object.stateSet(self, state: int) -> None
 |      Set the PETSc object state.
 |      
 |              Logically collective.
 |      
 |              See Also
 |              --------
 |              stateIncrease, stateGet, petsc.PetscObjectStateSet
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:471
 |  
 |  viewFromOptions(...)
 |      Object.viewFromOptions(self, name: str, objpre: Object | None = None) -> None
 |      View the object via command line options.
 |      
 |              Collective.
 |      
 |              Parameters
 |              ----------
 |              name
 |                  The command line option.
 |              objpre
 |                  Optional object that provides prefix.
 |      
 |              See Also
 |              --------
 |              petsc_options, petsc.PetscObjectViewFromOptions
 |      
 |              
 |      Source code at petsc4py/PETSc/Object.pyx:164
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from Object:
 |  
 |  classid
 |      Object.classid: int
 |      The class identifier.
 |      Source code at petsc4py/PETSc/Object.pyx:558
 |  
 |  comm
 |      Object.comm: Comm
 |      The object communicator.
 |      Source code at petsc4py/PETSc/Object.pyx:545
 |  
 |  fortran
 |      Object.fortran: int
 |      Fortran handle.
 |      Source code at petsc4py/PETSc/Object.pyx:588
 |  
 |  handle
 |      Object.handle: int
 |      Handle for ctypes support.
 |      Source code at petsc4py/PETSc/Object.pyx:580
 |  
 |  id
 |      Object.id: int
 |      The object identifier.
 |      Source code at petsc4py/PETSc/Object.pyx:563
 |  
 |  klass
 |      Object.klass: str
 |      The class name.
 |      Source code at petsc4py/PETSc/Object.pyx:568
 |  
 |  name
 |      Object.name: str
 |      The object name.
 |      Source code at petsc4py/PETSc/Object.pyx:550
 |  
 |  prefix
 |      Object.prefix: str
 |      Options prefix.
 |      Source code at petsc4py/PETSc/Object.pyx:537
 |  
 |  refcount
 |      Object.refcount: int
 |      Reference count.
 |      Source code at petsc4py/PETSc/Object.pyx:573
 |  
 |  type
 |      Object.type: str
 |      Object type.
 |      Source code at petsc4py/PETSc/Object.pyx:529
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from Object:
 |  
 |  __hash__ = None

"""
