nx = 1280
ny = 1024
cgh = cudacgh_allocate(nx, ny)

tic
cudacgh_initialize, cgh

for i=0,19 do $
  cudacgh_addtrap, cgh, $
	[100.*(randomu(seed, 3) - 0.5), 1., 2.*!pi*randomu(seed)]

b = cudacgh_getphase(cgh)
toc

cudacgh_free, cgh
