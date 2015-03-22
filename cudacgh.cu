//
// cudacgh
//
// CUDA-accelerated calculation of computer-generated holograms
// for holographic optical trapping.
//
// MODIFICATION HISTORY:
// 03/22/2015 Written by David G. Grier, New York University
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
  float *phi;
  size_t width;
  size_t height;
  size_t len;
  size_t nbytes;
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
    cgh.phi[i] = atan2f(cgh.psir[i], cgh.psii[i]) + M_PI;
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
  IDL_VPTR idl_cgh;
  char *pcgh;
  CGH_BUFFER cgh;

  cgh.width  = IDL_LongScalar(argv[0]);
  cgh.height = IDL_LongScalar(argv[1]);
  cgh.len = cgh.width * cgh.height;
  cgh.nbytes = cgh.len * sizeof(float);
  CudaSafeCall( cudaMalloc((void **) &cgh.x, cgh.nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.y, cgh.nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.psir, cgh.nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.psii, cgh.nbytes) );
  CudaSafeCall( cudaMalloc((void **) &cgh.phi, cgh.nbytes) );

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
  
  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  CudaSafeCall( cudaMemset(cgh.psii, 0, cgh.nbytes) );
  CudaSafeCall( cudaMemset(cgh.psir, 0, cgh.nbytes) );
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
  // use actual parameters from IDL
  // cal.kx = cgh.width/2.;
  // cal.ky = cgh.height/2.;
  // cal.q = 2.*M_PI/cgh.width;
  // cal.aspect_ratio = 1.;
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
  IDL_MEMINT dim[2];
  IDL_VPTR idl_phi;
  char *pd;

  IDL_VarGetData(argv[0], &n, &pcgh, TRUE);
  memcpy(&cgh, pcgh, sizeof(CGH_BUFFER));

  getphase<<<(cgh.len + 255)/256, 256>>>(cgh);
  CudaCheckError();
  
  dim[0] = cgh.width;
  dim[1] = cgh.height;
  pd = IDL_MakeTempArray(IDL_TYP_FLOAT, 2, dim, IDL_ARR_INI_NOP, &idl_phi);
  CudaSafeCall( cudaMemcpy(pd, cgh.phi, cgh.nbytes, cudaMemcpyDeviceToHost) );
  
  return idl_phi;
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
  };

  static IDL_SYSFUN_DEF2 procedure_addr[] = {
    { (IDL_SYSRTN_GENERIC) cudacgh_free,
      (char *) "CUDACGH_FREE", 1, 1, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_initialize,
      (char *) "CUDACGH_INITIALIZE", 1, 1, 0, 0 },
    { (IDL_SYSRTN_GENERIC) cudacgh_addtrap,
      (char *) "CUDACGH_ADDTRAP", 3, 3, 0, 0 }
  };
  
  nfcns = IDL_CARRAY_ELTS(function_addr);
  npros = IDL_CARRAY_ELTS(procedure_addr);
  status = IDL_SysRtnAdd(function_addr, TRUE, nfcns);
  status |= IDL_SysRtnAdd(procedure_addr, FALSE, npros);

  return status;
}
