//
// cudacgh
//
// CUDA-accelerated calculation of computer-generated holograms
// for holographic optical trapping.
//
// MODIFICATION HISTORY:
// 03/22/2015 Written by David G. Grier, New York University
// 07/25/2015 DGG Implemented CUDACGH_GETFIELD()
//
// Copyright (c) 2015 David G. Grier
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// IDL Support
#include "idl_export.h"

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK
#include "cudasafe.h"

typedef struct cgh_buffer {
  float *x;
  float *y;
  float *psir;
  float *psii;
  unsigned char *phi;
  size_t width;
  size_t len;
} CGH_BUFFER;

typedef struct cgh_calibration {
  float kx;
  float ky;
  float q;
  float aspect_ratio;
} CGH_CALIBRATION;

typedef struct cgh_trap {
  float x;
  float y;
  float z;
  float alpha;
  float phi;
} CGH_TRAP;

//
// addtrap
//
__global__ void addtrap(CGH_BUFFER cgh,
			CGH_CALIBRATION cal,
			CGH_TRAP p)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < cgh.len) {
    cgh.x[i] = ((float) (i % cgh.width)) - cal.kx;
    cgh.y[i] = ((float) (i / cgh.width)) - cal.ky;
    cgh.x[i] = cal.q * cgh.x[i];
    cgh.y[i] = cal.q * cal.aspect_ratio * cgh.y[i];
    cgh.x[i] = p.phi + p.x * cgh.x[i] + p.y * cgh.y[i]
      + p.z * (cgh.x[i] * cgh.x[i] + cgh.y[i] * cgh.y[i]);
    cgh.psir[i] = cgh.psir[i] + p.alpha * cosf(cgh.x[i]);
    cgh.psii[i] = cgh.psii[i] + p.alpha * sinf(cgh.x[i]);
  }
}

//
// getphase
//
__global__ void getphase(CGH_BUFFER cgh)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < cgh.len)
    cgh.phi[i] = (unsigned char) (127.5*(atan2f(cgh.psir[i], cgh.psii[i])/M_PI + 1.));
}

//
// setbackground
//
__global__ void setbackground(int n, float *psir, float *psii)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < n) {
    psii[i] = sinf(psir[i]);
    psir[i] = cosf(psir[i]);
  }
}

//
// cgh = cudacgh_allocate(width, height)
//
// Allocate memory for the GPU buffers in the CGH_BUFFER structure.
// Returns a "blind" array of BYTE to IDL which serves as an
// identifier for the CUDA CGH calculation stack.
//
extern "C" IDL_VPTR IDL_CDECL cudacgh_allocate(int argc, IDL_VPTR argv[])
{
  CGH_BUFFER cgh;
  size_t height;
  size_t nbytes;
  IDL_VPTR idl_cgh;
  char *pcgh;

  cgh.width  = IDL_LongScalar(argv[0]);
  height = IDL_LongScalar(argv[1]);
  cgh.len = cgh.width * height;
  nbytes = cgh.len * sizeof(float);
  CudaSafeCall( cudaMalloc((void **) &cgh.x, nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.y, nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.psir, nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.psii, nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.phi, cgh.len) );

  pcgh = IDL_MakeTempVector(IDL_TYP_BYTE, sizeof(CGH_BUFFER),
			    IDL_ARR_INI_NOP, &idl_cgh);
  memcpy(pcgh, &cgh, sizeof(CGH_BUFFER));
  
  return idl_cgh;
}

//
// cudacgh_initialize, cgh
//
// Set the field in the SLM plane to zero.
//
extern "C" void IDL_CDECL cudacgh_initialize(int argc, IDL_VPTR argv[])
{
  IDL_MEMINT n;
  char *pcgh;
  CGH_BUFFER cgh;
  size_t nbytes;
  
  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  nbytes = cgh.len * sizeof(float);

  if ((argc == 2) &&
      (argv[1]->type == IDL_TYP_FLOAT) &&
      (argv[1]->flags & IDL_V_ARR) &&
      (argv[1]->value.arr->arr_len == nbytes)) {
    CudaSafeCall( cudaMemcpy(cgh.psir, argv[1]->value.arr->data, nbytes,
			     cudaMemcpyHostToDevice) );
    setbackground<<<(cgh.len + 255)/256, 256>>>(nbytes, cgh.psir, cgh.psii);
    CudaCheckError();
  } else {
    CudaSafeCall( cudaMemset(cgh.psii, 0, nbytes) );
    CudaSafeCall( cudaMemset(cgh.psir, 0, nbytes) );
  } 
}

//
// cudacgh_free, cgh
//
// Deallocate GPU buffers in the CGH_BUFFER structure
//
extern "C" void IDL_CDECL cudacgh_free(int argc, IDL_VPTR argv[])
{
  IDL_MEMINT n;
  char *pcgh;
  CGH_BUFFER cgh;
  
  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  CudaSafeCall( cudaFree(cgh.x) );
  CudaSafeCall( cudaFree(cgh.y) );
  CudaSafeCall( cudaFree(cgh.psir) );
  CudaSafeCall( cudaFree(cgh.psii) );
  CudaSafeCall( cudaFree(cgh.phi) );
}

//
// cudacgh_addtrap, cgh, cal, p
//
// Add the field due to a specified trap to the current
// field in the CGH structure.
//
extern "C" void IDL_CDECL cudacgh_addtrap(int argc, IDL_VPTR argv[])
{
  IDL_MEMINT n;
  char *pcgh;
  CGH_BUFFER cgh;
  CGH_CALIBRATION cal;
  char *pdata;
  CGH_TRAP p;

  // CGH_BUFFER structure  
  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  // CGH calibration constants
  IDL_ENSURE_ARRAY(argv[2]);
  if ((argv[1]->value.arr->n_elts != 4) ||
      (argv[1]->value.arr->arr_len != sizeof(CGH_CALIBRATION))) {
    IDL_Message(IDL_M_NAMED_GENERIC, IDL_MSG_INFO,
		"Not valid calibration settings.  Skipping.");
    return;
  }
  IDL_VarGetData(argv[1], &n, &pdata, TRUE);
  memcpy(&cal, pdata, sizeof(CGH_CALIBRATION));
  
  // Trap position
  IDL_ENSURE_ARRAY(argv[2]);
  if ((argv[2]->value.arr->n_elts != 5) ||
      (argv[2]->value.arr->arr_len != sizeof(CGH_TRAP))) {
    IDL_Message(IDL_M_NAMED_GENERIC, IDL_MSG_INFO,
		"Not a valid trap description.  Skipping.");
    return;
  }
  IDL_VarGetData(argv[2], &n, &pdata, TRUE);
  memcpy(&p, pdata, sizeof(CGH_TRAP));
  
  addtrap<<<(cgh.len + 255)/256, 256>>>(cgh, cal, p);
  CudaCheckError();
}

//
// phi = cudacgh_getphase(cgh)
//
// Returns the floating-point hologram associated with
// the current field in the SLM plane.
//
extern "C" IDL_VPTR IDL_CDECL cudacgh_getphase(int argc, IDL_VPTR argv[])
{
  IDL_MEMINT n;
  char *pcgh;
  CGH_BUFFER cgh;
  IDL_MEMINT dim[IDL_MAX_ARRAY_DIM];
  IDL_VPTR idl_phi;
  char *pd;

  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  getphase<<<(cgh.len + 255)/256, 256>>>(cgh);
  CudaCheckError();
  
  dim[0] = cgh.width;
  dim[1] = cgh.len/cgh.width; // height
  pd = IDL_MakeTempArray(IDL_TYP_BYTE, 2, dim, IDL_ARR_INI_NOP, &idl_phi);
  CudaSafeCall( cudaMemcpy(pd, cgh.phi, cgh.len, cudaMemcpyDeviceToHost) );
  
  return idl_phi;
}

//
// field = cudacgh_getfield(cgh)
//
// Returns the complex-valued field in the SLM plane.
//
extern "C" IDL_VPTR IDL_CDECL cudacgh_getfield(int argc, IDL_VPTR argv[])
{
  IDL_MEMINT n, len;
  char *pcgh;
  CGH_BUFFER cgh;
  IDL_MEMINT dim[IDL_MAX_ARRAY_DIM];
  IDL_VPTR idl_field;
  char *pd;

  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  dim[0] = cgh.width;
  dim[1] = cgh.len/cgh.width; // height
  dim[2] = 2;
  pd = IDL_MakeTempArray(IDL_TYP_FLOAT, 3, dim, IDL_ARR_INI_ZERO, &idl_field);
  
  len = cgh.len * sizeof(float);
  CudaSafeCall( cudaMemcpy(pd    , cgh.psir, len, cudaMemcpyDeviceToHost) );
  CudaSafeCall( cudaMemcpy(pd+len, cgh.psii, len, cudaMemcpyDeviceToHost) );
      
  return idl_field;
}

//
// IDL_Load
//
extern "C" int IDL_Load(void)
{
  int status;
  int nfcns, npros;

  static IDL_SYSFUN_DEF2 function_addr[] = {
    { (IDL_SYSRTN_GENERIC) cudacgh_allocate, 
      (char *) "CUDACGH_ALLOCATE", 2, 2, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_getphase,
      (char *) "CUDACGH_GETPHASE", 1, 1, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_getfield,
      (char *) "CUDACGH_GETFIELD", 1, 1, 0, 0 },
  };

  static IDL_SYSFUN_DEF2 procedure_addr[] = {
    { (IDL_SYSRTN_GENERIC) cudacgh_free,
      (char *) "CUDACGH_FREE", 1, 1, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_initialize,
      (char *) "CUDACGH_INITIALIZE", 1, 2, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_addtrap,
      (char *) "CUDACGH_ADDTRAP", 3, 3, 0, 0 }
  };
  
  nfcns = IDL_CARRAY_ELTS(function_addr);
  npros = IDL_CARRAY_ELTS(procedure_addr);
  status = IDL_SysRtnAdd(function_addr, TRUE, nfcns);
  status |= IDL_SysRtnAdd(procedure_addr, FALSE, npros);

  return status;
}
