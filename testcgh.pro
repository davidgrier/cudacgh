nx = 1280
ny = 1024
cal = [nx/2., ny/2., 2.*!pi/nx, 1.]
cgh = cudacgh_allocate(nx, ny)

tic
cudacgh_initialize, cgh

for i=0,19 do $
  cudacgh_addtrap, cgh, cal, $
	[100.*(randomu(seed, 3) - 0.5), 1., 2.*!pi*randomu(seed)]

b = cudacgh_getphase(cgh)
toc

cudacgh_free, cgh
