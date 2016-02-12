## Introduction ##
dedisp uses a GPU to perform the computationally intensive task of computing the incoherent dedispersion transform, a frequent operation in signal processing and time-domain radio astronomy. It currently uses NVIDIA's CUDA platform and supports all CUDA-capable GPUs.

For a detailed discussion of how the library implements dedispersion on the GPU, see [Barsdell et al. 2012](http://adsabs.harvard.edu/abs/2012arXiv1201.5380B). If you use the library in your work, please consider citing this paper.

## Features ##
  * High performance: up to **10x speed-up** over efficient quad-core CPU implementation
  * **Pure C interface** allows easy integration into existing C/C++/Fortran/Python etc. codes
  * Accepts input time series sampled with **1, 2, 4, 8, 16 or 32 bits per sample**
  * Can produce dedispersed time series sampled with **8, 16 or 32 bits per sample**
  * **DM trials can be generated** by the library **or supplied by the user**
  * Accepts a channel **'killmask'** for skipping bad channels
  * **Adaptive time resolution** (aka _time-scrunching_, _binning_) for further speed gains
  * Input and output data can be passed **from the host or directly from the GPU**
  * Extended **'advanced' and 'guru' interfaces** allow arbitrary data strides and DM gulping
  * Optional **C++ wrapper** for convenient object-oriented syntax

## Downloading ##
The complete source distribution can be checked-out via the subversion repository under the [Source](http://code.google.com/p/dedisp/source/checkout) tab. Directly downloadable releases may be added in the future.

## Example ##
```
#include <stdlib.h>
#include <stdio.h>
#include <dedisp.h>

int main(int argc, char* argv[])
{
	int          device_idx  = 0;
	dedisp_size  nchans      = 1024;
	dedisp_float dt          = 64e-6;     // s
	dedisp_float f0          = 1581.8;    // MHz
	dedisp_float df          = 0.39062;   // MHz
	dedisp_float dm_start    = 0.0;       // pc cm^-3
	dedisp_float dm_end      = 1000.0;    // pc cm^-3
	dedisp_float pulse_width = 40;        // ms
	dedisp_float dm_tol      = 1.25;
	dedisp_size  nsamps      = 60 / dt;
	dedisp_size  in_nbits    = 2;
	dedisp_size  out_nbits   = 8;
	
	dedisp_plan  plan;
	dedisp_error error;
	dedisp_size  dm_count;
	dedisp_size  max_delay;
	dedisp_size  nsamps_computed;
	dedisp_byte* input  = 0;
	dedisp_byte* output = 0;
	
	// Load input data from somewhere
	input = malloc(nsamps * nchans * in_nbits/8);
	// ...
	
	// Initialise the GPU
	error = dedisp_set_device(device_idx);
	if( error != DEDISP_NO_ERROR ) {
		printf("ERROR: Could not set GPU device: %s\n",
		       dedisp_get_error_string(error));
		return -1;
	}
	
	// Create a dedispersion plan
	error = dedisp_create_plan(&plan, nchans, dt, f0, df);
	if( error != DEDISP_NO_ERROR ) {
		printf("\nERROR: Could not create dedispersion plan: %s\n",
		       dedisp_get_error_string(error));
		return -1;
	}
	
	// Generate a list of dispersion measures for the plan
	error = dedisp_generate_dm_list(plan, dm_start, dm_end, pulse_width, dm_tol);
	if( error != DEDISP_NO_ERROR ) {
		printf("\nERROR: Failed to generate DM list: %s\n",
		       dedisp_get_error_string(error));
		return -1;
	}
	
	// Find the parameters that determine the output size
	dm_count = dedisp_get_dm_count(plan);
	max_delay = dedisp_get_max_delay(plan);
	nsamps_computed = nsamps - max_delay;
	
	// Allocate space for the output data
	output = malloc(nsamps_computed * dm_count * out_nbits/8);
	
	// Compute the dedispersion transform on the GPU
	error = dedisp_execute(plan, nsamps,
	                       input, in_nbits,
	                       output, out_nbits,
	                       DEDISP_USE_DEFAULT);
	if( error != DEDISP_NO_ERROR ) {
		printf("\nERROR: Failed to execute dedispersion plan: %s\n",
		       dedisp_get_error_string(error));
		return -1;
	}
	
	// Do something with the output data
	// ...
	
	// Clean up
	free(output);
	free(input);
	dedisp_destroy_plan(plan);
	printf("Dedispersion successful.\n");
	return 0;
}
```