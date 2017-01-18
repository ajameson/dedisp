# dedisp
This repositry is derived from Ben Barsdell's original GPU De-dedispersion library (code.google.com/p/dedisp)

Installation Instructions:

  1.  git clone https://github.com/ajameson/dedisp.git
  2.  Update Makefile.inc with your CUDA path, Install Dir and GPU architecture
  3.  make && make install
  
  This will build a shared object library named libdedisp.so which is a prerequisite for Heimdall
