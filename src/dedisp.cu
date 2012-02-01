/*
 *  Copyright 2012 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
  This file contains the boring boiler-plate code to manage the library.
*/

//#define DEDISP_DEBUG
//#define DEDISP_BENCHMARK

#include <dedisp.h>

#include <vector>
#include <algorithm> // For std::fill

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef DEDISP_BENCHMARK
#include <fstream>
#endif

// TODO: Remove these when done benchmarking
// -----------------------------------------
#if defined(DEDISP_BENCHMARK)
#include <iostream>
using std::cout;
using std::endl;
#include "stopwatch.hpp"
#endif
// -----------------------------------------

#include "gpu_memory.hpp"
#include "transpose.hpp"

#define DEDISP_DEFAULT_GULP_SIZE 131072

// Note: The implementation of the sub-band algorithm is a prototype only
//         Enable at your own risk!
//#define USE_SUBBAND_ALGORITHM
#define DEDISP_DEFAULT_SUBBAND_SIZE 32

// TODO: Make sure this doesn't limit GPU constant memory
//         available to users.
#define DEDISP_MAX_NCHANS 8192
// Internal word type used for transpose and dedispersion kernel
typedef unsigned int dedisp_word;
// Note: This must be included after the above #define and typedef
#include "kernels.cuh"

// Define plan structure
struct dedisp_plan_struct {
	// Size parameters
	dedisp_size  dm_count;
	dedisp_size  nchans;
	dedisp_size  max_delay;
	dedisp_size  gulp_size;
	// Physical parameters
	dedisp_float dt;
	dedisp_float f0;
	dedisp_float df;
	// Host arrays
	std::vector<dedisp_float> dm_list;
	std::vector<dedisp_float> delay_table;
	std::vector<dedisp_bool>  killmask;
	// Device arrays
	thrust::device_vector<dedisp_float> d_dm_list;
	thrust::device_vector<dedisp_float> d_delay_table;
	thrust::device_vector<dedisp_bool> d_killmask;
	//StreamType stream;
};

// Private helper functions
// ------------------------
template<typename T>
T min(T a, T b) { return a<b ? a : b; }
unsigned long div_round_up(unsigned long a, unsigned long b) {
	return (a-1) / b + 1;
}

// Internal abstraction for errors
#if defined(DEDISP_DEBUG) && DEDISP_DEBUG
#define throw_error(error) do {                                         \
	printf("An error occurred within dedisp on line %d of %s: %s",      \
	       __LINE__, __FILE__, dedisp_get_error_string(error));         \
	return error; } while(0)
#else
#define throw_error(error) return error
#endif // DEDISP_DEBUG
/*
dedisp_error throw_error(dedisp_error error) {
	// Note: Could, e.g., put an error callback in here
	return error;
}
*/
// ------------------------

// Public functions
// ----------------
dedisp_error dedisp_create_plan(dedisp_plan* plan_,
                                dedisp_size  nchans,
                                dedisp_float dt,
                                dedisp_float f0,
                                dedisp_float df)
{
	// Initialise to NULL for safety
	*plan_ = 0;
	
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	
	int device_idx;
	cudaGetDevice(&device_idx);
	
	// Check for parameter errors
	if( nchans > DEDISP_MAX_NCHANS ) {
		throw_error(DEDISP_NCHANS_EXCEEDS_LIMIT);
	}
	
	dedisp_plan plan = new dedisp_plan_struct();
	if( !plan ) {
		throw_error(DEDISP_MEM_ALLOC_FAILED);
	}
	
	plan->dm_count      = 0;
	plan->nchans        = nchans;
	plan->gulp_size     = DEDISP_DEFAULT_GULP_SIZE;
	plan->max_delay     = 0;
	plan->dt            = dt;
	plan->f0            = f0;
	plan->df            = df;
	//plan->stream        = 0;
	
	// Generate delay table and copy to device memory
	// Note: The DM factor is left out and applied during dedispersion
	plan->delay_table.resize(plan->nchans);
	generate_delay_table(&plan->delay_table[0], plan->nchans, dt, f0, df);
	try {
		plan->d_delay_table.resize(plan->nchans);
	}
	catch(...) {
		dedisp_destroy_plan(plan);
		throw_error(DEDISP_MEM_ALLOC_FAILED);
	}
	try {
		plan->d_delay_table = plan->delay_table;
	}
	catch(...) {
		dedisp_destroy_plan(plan);
		throw_error(DEDISP_MEM_COPY_FAILED);
	}
	
	// Initialise the killmask
	plan->killmask.resize(plan->nchans, (dedisp_bool)true);
	try {
		plan->d_killmask.resize(plan->nchans);
	}
	catch(...) {
		dedisp_destroy_plan(plan);
		throw_error(DEDISP_MEM_ALLOC_FAILED);
	}
	dedisp_error err = dedisp_set_killmask(plan, (dedisp_bool*)0);
	if( err != DEDISP_NO_ERROR ) {
		dedisp_destroy_plan(plan);
		throw_error(err);
	}
	
	*plan_ = plan;
	
	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_gulp_size(dedisp_plan plan,
                                  dedisp_size gulp_size) {
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->gulp_size = gulp_size;
	return DEDISP_NO_ERROR;
}
dedisp_size dedisp_get_gulp_size(dedisp_plan plan) {
	return plan->gulp_size;
}

dedisp_error dedisp_set_dm_list(dedisp_plan plan,
                                const dedisp_float* dm_list,
                                dedisp_size count)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	if( !dm_list ) {
		throw_error(DEDISP_INVALID_POINTER);
	}
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	
	plan->dm_count = count;
	plan->dm_list.assign(dm_list, dm_list+count);
	
	// Copy to the device
	try {
		plan->d_dm_list.resize(plan->dm_count);
	}
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	try {
		plan->d_dm_list = plan->dm_list;
	}
	catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
	
	// Calculate the maximum delay and store it in the plan
	plan->max_delay = (dedisp_size)(plan->dm_list[plan->dm_count-1] *
	                                plan->delay_table[plan->nchans-1] + 0.5);
	
	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_generate_dm_list(dedisp_plan plan,
                                     dedisp_float dm_start, dedisp_float dm_end,
                                     dedisp_float ti, dedisp_float tol)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	
	// Generate the DM list (on the host)
	plan->dm_list.clear();
	generate_dm_list(plan->dm_list,
					 dm_start, dm_end,
					 plan->dt, ti, plan->f0, plan->df,
					 plan->nchans, tol);
	plan->dm_count = plan->dm_list.size();
	
	// Allocate device memory for the DM list
	try {
		plan->d_dm_list.resize(plan->dm_count);
	}
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	try {
		plan->d_dm_list = plan->dm_list;
	}
	catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
	
	// Calculate the maximum delay and store it in the plan
	plan->max_delay = dedisp_size(plan->dm_list[plan->dm_count-1] *
								  plan->delay_table[plan->nchans-1] + 0.5);
	
	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_device(int device_idx) {
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	
	cudaError_t error = cudaSetDevice(device_idx);
	// Note: cudaErrorInvalidValue isn't a documented return value, but
	//         it still gets returned :/
	if( cudaErrorInvalidDevice == error ||
		cudaErrorInvalidValue == error )
		throw_error(DEDISP_INVALID_DEVICE_INDEX);
	else if( cudaErrorSetOnActiveProcess == error )
		throw_error(DEDISP_DEVICE_ALREADY_SET);
	else if( cudaSuccess != error )
		throw_error(DEDISP_UNKNOWN_ERROR);
	else
		return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_killmask(dedisp_plan plan, const dedisp_bool* killmask)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	if( 0 != killmask ) {
		// Copy killmask to plan (both host and device)
		plan->killmask.assign(killmask, killmask + plan->nchans);
		try {
			plan->d_killmask = plan->killmask;
		}
		catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
	}
	else {
		// Set the killmask to all true
		std::fill(plan->killmask.begin(), plan->killmask.end(), (dedisp_bool)true);
		thrust::fill(plan->d_killmask.begin(), plan->d_killmask.end(),
		             (dedisp_bool)true);
	}
	return DEDISP_NO_ERROR;
}
/*
dedisp_plan dedisp_set_stream(dedisp_plan plan, StreamType stream)
{
	plan->stream = stream;
	return plan;
}
*/

// Getters
// -------
dedisp_size         dedisp_get_max_delay(const dedisp_plan plan) {
	return plan->max_delay;
}
dedisp_size         dedisp_get_channel_count(const dedisp_plan plan) {
	return plan->nchans;
}
dedisp_size         dedisp_get_dm_count(const dedisp_plan plan) {
	return plan->dm_count;
}
const dedisp_float* dedisp_get_dm_list(const dedisp_plan plan) {
	return &plan->dm_list[0];
}
const dedisp_bool*  dedisp_get_killmask(const dedisp_plan plan) {
	return &plan->killmask[0];
}
dedisp_float        dedisp_get_dt(const dedisp_plan plan) {
	return plan->dt;
}
dedisp_float        dedisp_get_f0(const dedisp_plan plan) {
	return plan->f0;
}
dedisp_float        dedisp_get_df(const dedisp_plan plan) {
	return plan->df;
}

// Warning: Big mother function
dedisp_error dedisp_execute_guru(const dedisp_plan  plan,
                                 dedisp_size        nsamps,
                                 const dedisp_byte* in,
                                 dedisp_size        in_nbits,
                                 dedisp_size        in_stride,
                                 dedisp_byte*       out,
                                 dedisp_size        out_nbits,
                                 dedisp_size        out_stride,
                                 dedisp_size        first_dm_idx,
                                 dedisp_size        dm_count,
                                 unsigned           flags)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	
	enum {
		BITS_PER_BYTE  = 8,
		BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
	};
	
	dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
	                                                BITS_PER_BYTE);
	
	if( 0 == in || 0 == out ) {
		throw_error(DEDISP_INVALID_POINTER);
	}
	// Note: Must be careful with integer division
	if( in_stride < plan->nchans*in_nbits/(sizeof(dedisp_byte)*BITS_PER_BYTE) ||
	    out_stride < (nsamps - plan->max_delay)*out_bytes_per_sample ) {
		throw_error(DEDISP_INVALID_STRIDE);
	}
	if( 0 == plan->dm_count ) {
		throw_error(DEDISP_NO_DM_LIST_SET);
	}
	if( nsamps < plan->max_delay ) {
		throw_error(DEDISP_TOO_FEW_NSAMPS);
	}
	
	// Check for valid synchronisation flags
	if( flags & DEDISP_ASYNC && flags & DEDISP_WAIT ) {
		throw_error(DEDISP_INVALID_FLAG_COMBINATION);
	}
	
	// Check for valid nbits values
	if( in_nbits  != 1 &&
	    in_nbits  != 2 &&
	    in_nbits  != 4 &&
	    in_nbits  != 8 &&
	    in_nbits  != 16 &&
	    in_nbits  != 32 ) {
		throw_error(DEDISP_UNSUPPORTED_IN_NBITS);
	}
	if( out_nbits != 8 &&
	    out_nbits != 16 &&
	    out_nbits != 32 ) {
		throw_error(DEDISP_UNSUPPORTED_OUT_NBITS);
	}
	
	bool using_host_memory;
	if( flags & DEDISP_HOST_POINTERS && flags & DEDISP_DEVICE_POINTERS ) {
		throw_error(DEDISP_INVALID_FLAG_COMBINATION);
	}
	else {
		using_host_memory = !(flags & DEDISP_DEVICE_POINTERS);
	}
	
	// Copy the lookup tables to constant memory on the device
	if( !copy_device_to_symbol("c_delay_table",
	                           thrust::raw_pointer_cast(&plan->d_delay_table[0]),
	                           plan->nchans) ) {
		throw_error(DEDISP_MEM_COPY_FAILED);
	}
	if( !copy_device_to_symbol("c_killmask",
	                           thrust::raw_pointer_cast(&plan->d_killmask[0]),
	                           plan->nchans) ) {
		throw_error(DEDISP_MEM_COPY_FAILED);
	}
	
	// Compute the problem decomposition
	dedisp_size nsamps_computed = nsamps - plan->max_delay;
	// Specify the maximum gulp size
	dedisp_size nsamps_computed_gulp_max;
	if( using_host_memory ) {
		nsamps_computed_gulp_max = min(plan->gulp_size, nsamps_computed);
	}
	else {
		// Just do it in one gulp if given device pointers
		nsamps_computed_gulp_max = nsamps_computed;
	}
	
	// Just to be sure
	if( nsamps_computed_gulp_max < plan->max_delay ) {
		throw_error(DEDISP_TOO_FEW_NSAMPS);
	}
	
	// Compute derived counts for maximum gulp size
	dedisp_size nsamps_gulp_max = nsamps_computed_gulp_max + plan->max_delay;
	dedisp_size chans_per_word  = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
	dedisp_size nchan_words     = plan->nchans / chans_per_word;
	
	// We use words for processing but allow arbitrary byte strides, which are
	//   not necessarily friendly.
	bool friendly_in_stride = (0 == in_stride % BYTES_PER_WORD);
	
	// Note: If desired, this could be rounded up, e.g., to a power of 2
	dedisp_size in_buf_stride_words      = nchan_words;
	dedisp_size in_count_gulp_max        = nsamps_gulp_max * in_buf_stride_words;
	
	dedisp_size nsamps_padded_gulp_max   = div_round_up(nsamps_computed_gulp_max,
	                                                    DEDISP_SAMPS_PER_THREAD)
		* DEDISP_SAMPS_PER_THREAD + plan->max_delay;
	dedisp_size in_count_padded_gulp_max = 
		nsamps_padded_gulp_max * in_buf_stride_words;
	dedisp_size out_stride_gulp_samples  = nsamps_computed_gulp_max;
	dedisp_size out_stride_gulp_bytes    = 
		out_stride_gulp_samples * out_bytes_per_sample;
	dedisp_size out_count_gulp_max       = out_stride_gulp_bytes * dm_count;
	
	// Organise device memory pointers
	// -------------------------------
	const dedisp_word* d_in = 0;
	dedisp_word*       d_transposed = 0;
	dedisp_byte*       d_out = 0;
	thrust::device_vector<dedisp_word> d_in_buf;
	thrust::device_vector<dedisp_word> d_transposed_buf;
	thrust::device_vector<dedisp_byte> d_out_buf;
	// Allocate temporary buffers on the device where necessary
	if( using_host_memory || !friendly_in_stride ) {
		try { d_in_buf.resize(in_count_gulp_max); }
		catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
		d_in = thrust::raw_pointer_cast(&d_in_buf[0]);
	}
	else {
		d_in = (dedisp_word*)in;
	}
	if( using_host_memory ) {
		try { d_out_buf.resize(out_count_gulp_max); }
		catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
		d_out = thrust::raw_pointer_cast(&d_out_buf[0]);
	}
	else {
		d_out = out;
	}
	try { d_transposed_buf.resize(in_count_padded_gulp_max); }
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	d_transposed = thrust::raw_pointer_cast(&d_transposed_buf[0]);
	// -------------------------------
	
#ifdef USE_SUBBAND_ALGORITHM
	
	dedisp_size sb_size           = DEDISP_DEFAULT_SUBBAND_SIZE;
	// Note: Setting these two parameters equal should balance the two steps of
	//         the sub-band algorithm.
	dedisp_size dm_size           = sb_size; // Ndm'
	
	dedisp_size sb_count          = plan->nchans / sb_size;
	dedisp_size nom_dm_count      = dm_count / dm_size;
	
	thrust::device_vector<dedisp_word> d_intermediate_buf;
	try { d_intermediate_buf.resize(nsamps_padded_gulp_max * sb_count
	                                * nom_dm_count); }
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	dedisp_word* d_intermediate = thrust::raw_pointer_cast(&d_intermediate_buf[0]);
	
#endif //  USE_SUBBAND_ALGORITHM
	
	// TODO: Eventually re-implement streams
	cudaStream_t stream = 0;//(cudaStream_t)plan->stream;
	
#ifdef DEDISP_BENCHMARK
	Stopwatch copy_to_timer;
	Stopwatch copy_from_timer;
	Stopwatch transpose_timer;
	Stopwatch kernel_timer;
#endif
	
	// Gulp loop
	for( dedisp_size gulp_samp_idx=0; 
	     gulp_samp_idx<nsamps_computed; 
	     gulp_samp_idx+=nsamps_computed_gulp_max ) {
		
		dedisp_size nsamps_computed_gulp = min(nsamps_computed_gulp_max,
		                                       nsamps_computed-gulp_samp_idx);
		dedisp_size nsamps_gulp          = nsamps_computed_gulp + plan->max_delay;
		dedisp_size nsamps_padded_gulp   = div_round_up(nsamps_computed_gulp,
		                                                DEDISP_SAMPS_PER_THREAD)
			* DEDISP_SAMPS_PER_THREAD + plan->max_delay;
		
#ifdef DEDISP_BENCHMARK
		copy_to_timer.start();
#endif
		// Copy the input data from host to device if necessary
		if( using_host_memory ) {
			// Allowing arbitrary byte strides means we must do a strided copy
			if( !copy_host_to_device_2d((dedisp_byte*)d_in,
			                            in_buf_stride_words * BYTES_PER_WORD,
			                            in + gulp_samp_idx*in_stride,
			                            in_stride,
			                            nchan_words * BYTES_PER_WORD,
			                            nsamps_gulp) ) {
				throw_error(DEDISP_MEM_COPY_FAILED);
			}
		}
		else if( !friendly_in_stride ) {
			// Device pointers with unfriendly stride
			if( !copy_device_to_device_2d((dedisp_byte*)d_in,
			                              in_buf_stride_words * BYTES_PER_WORD,
			                              in + gulp_samp_idx*in_stride,
			                              in_stride,
			                              nchan_words * BYTES_PER_WORD,
			                              nsamps_gulp) ) {
				throw_error(DEDISP_MEM_COPY_FAILED);
			}
		}
#ifdef DEDISP_BENCHMARK
		cudaThreadSynchronize();
		copy_to_timer.stop();
		transpose_timer.start();
#endif
		// Transpose the words in the input
		Transpose<dedisp_word> transpose;
		transpose.transpose(d_in,
		                    nchan_words, nsamps_gulp,
		                    in_buf_stride_words, nsamps_padded_gulp,
		                    d_transposed);
#ifdef DEDISP_BENCHMARK
		cudaThreadSynchronize();
		transpose_timer.stop();
		
		kernel_timer.start();
#endif
		
#ifdef USE_SUBBAND_ALGORITHM
		dedisp_size chan_stride       = 1;
		dedisp_size dm_stride         = dm_size;
		dedisp_size ostride           = nsamps_padded_gulp * sb_count;
		dedisp_size batch_size        = sb_count;
		dedisp_size batch_in_stride   = nsamps_padded_gulp * sb_size / chans_per_word;
		dedisp_size batch_dm_stride   = 0;
		dedisp_size batch_chan_stride = sb_size;
		dedisp_size batch_out_stride  = nsamps_padded_gulp;
		
		/* // Consistency checks
		if( (nom_dm_count-1)*dm_stride + (batch_size-1)*batch_dm_stride >= dm_count ) {
			throw std::runtime_error("DM STRIDES ARE INCONSISTENT");
		}
		if( (sb_size-1)*chan_stride + (batch_size-1)*batch_chan_stride >= plan->nchans ) {
			throw std::runtime_error("CHAN STRIDES ARE INCONSISTENT");
		}
		*/
		
		// Both steps
		if( !dedisperse(d_transposed,
		                nsamps_padded_gulp,
		                nsamps_computed_gulp,
		                in_nbits,
		                sb_size,
		                chan_stride,
		                thrust::raw_pointer_cast(&plan->d_dm_list[first_dm_idx]),
		                nom_dm_count,
		                dm_stride,
		                (dedisp_byte*)d_intermediate,
		                ostride,
		                32,//out_nbits,
		                batch_size,
		                batch_in_stride,
		                batch_dm_stride,
		                batch_chan_stride,
		                batch_out_stride) ) {
			throw_error(DEDISP_INTERNAL_GPU_ERROR);
		}
		
		batch_size = nom_dm_count;
		chan_stride       = sb_size;
		dm_stride         = 1;
		ostride           = out_stride_gulp_samples;
		batch_in_stride   = nsamps_padded_gulp * sb_count;
		batch_dm_stride   = 0;
		batch_chan_stride = 0;
		batch_out_stride  = out_stride_gulp_samples * dm_size;
		
		/* // Consistency checks
		if( (dm_size-1)*dm_stride + (batch_size-1)*batch_dm_stride >= dm_count ) {
			throw std::runtime_error("DM STRIDES ARE INCONSISTENT");
		}
		if( (sb_count-1)*chan_stride + (batch_size-1)*batch_chan_stride >= plan->nchans ) {
			throw std::runtime_error("CHAN STRIDES ARE INCONSISTENT");
		}
		*/
		
		if( !dedisperse(d_intermediate,
		                nsamps_padded_gulp,
		                nsamps_computed_gulp,
		                32,//in_nbits,
		                sb_count,
		                chan_stride,
		                thrust::raw_pointer_cast(&plan->d_dm_list[first_dm_idx]),
		                dm_size,
		                dm_stride,
		                d_out,
		                ostride,
		                out_nbits,
		                batch_size,
		                batch_in_stride,
		                batch_dm_stride,
		                batch_chan_stride,
		                batch_out_stride) ) {
			throw_error(DEDISP_INTERNAL_GPU_ERROR);
		}
#else // Use direct algorithm
		// Perform direct dedispersion
		if( !dedisperse(d_transposed,
		                nsamps_padded_gulp,
		                nsamps_computed_gulp,
		                in_nbits,
		                plan->nchans,
		                1,
		                thrust::raw_pointer_cast(&plan->d_dm_list[first_dm_idx]),
		                dm_count,
		                1,
		                d_out,
		                out_stride_gulp_samples,
		                out_nbits,
		                1, 0, 0, 0, 0) ) {
			throw_error(DEDISP_INTERNAL_GPU_ERROR);
		}

#endif // SB/direct algorithm

#ifdef DEDISP_BENCHMARK
		cudaThreadSynchronize();
		kernel_timer.stop();
#endif
		// Copy output back to host memory if necessary
		if( using_host_memory ) {
			dedisp_size gulp_samp_byte_idx = gulp_samp_idx * out_bytes_per_sample;
			dedisp_size nsamp_bytes_computed_gulp = nsamps_computed_gulp * out_bytes_per_sample;
#ifdef DEDISP_BENCHMARK
			copy_from_timer.start();
#endif
			copy_device_to_host_2d(out + gulp_samp_byte_idx,
			                       out_stride,
			                       d_out,
			                       out_stride_gulp_bytes,
			                       nsamp_bytes_computed_gulp,
			                       dm_count);
#ifdef DEDISP_BENCHMARK
			cudaThreadSynchronize();
			copy_from_timer.stop();
#endif
		}
		
	} // End of gulp loop
	
#ifdef DEDISP_BENCHMARK
	cout << "Copy to time:   " << copy_to_timer.getTime() << endl;
	cout << "Copy from time: " << copy_from_timer.getTime() << endl;
	cout << "Transpose time: " << transpose_timer.getTime() << endl;
	cout << "Kernel time:    " << kernel_timer.getTime() << endl;
	float total_time = copy_to_timer.getTime() + copy_from_timer.getTime() + transpose_timer.getTime() + kernel_timer.getTime();
	cout << "Total time:     " << total_time << endl;
	
	// Append the timing results to a log file
	std::ofstream perf_file("perf.log", std::ios::app);
	perf_file << copy_to_timer.getTime() << "\t"
	          << copy_from_timer.getTime() << "\t"
	          << transpose_timer.getTime() << "\t"
	          << kernel_timer.getTime() << "\t"
	          << total_time << endl;
	perf_file.close();
#endif
	
	if( !(flags & DEDISP_ASYNC) ) {
		cudaStreamSynchronize(stream);
	}
	
	// Phew!
	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_execute_adv(const dedisp_plan  plan,
                                dedisp_size        nsamps,
                                const dedisp_byte* in,
                                dedisp_size        in_nbits,
                                dedisp_size        in_stride,
                                dedisp_byte*       out,
                                dedisp_size        out_nbits,
                                dedisp_size        out_stride,
                                unsigned           flags)
{
	dedisp_size first_dm_idx = 0;
	dedisp_size dm_count = plan->dm_count;
	return dedisp_execute_guru(plan, nsamps,
	                           in, in_nbits, in_stride,
	                           out, out_nbits, out_stride,
	                           first_dm_idx, dm_count,
	                           flags);
}

// TODO: Consider having the user specify nsamps_computed instead of nsamps
dedisp_error dedisp_execute(const dedisp_plan  plan,
                            dedisp_size        nsamps,
                            const dedisp_byte* in,
                            dedisp_size        in_nbits,
                            dedisp_byte*       out,
                            dedisp_size        out_nbits,
                            unsigned           flags)
{
	
	enum {
		BITS_PER_BYTE = 8
	};
	
	// Note: The default out_stride is nsamps - plan->max_delay
	dedisp_size out_bytes_per_sample =
		out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
	
	// Note: Must be careful with integer division
	dedisp_size in_stride =
		plan->nchans * in_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
	dedisp_size out_stride = (nsamps - plan->max_delay) * out_bytes_per_sample;
	return dedisp_execute_adv(plan, nsamps,
	                          in, in_nbits, in_stride,
	                          out, out_nbits, out_stride,
	                          flags);
}

dedisp_error dedisp_sync(void)
{
	if( cudaThreadSynchronize() != cudaSuccess )
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	else
		return DEDISP_NO_ERROR;
}

void dedisp_destroy_plan(dedisp_plan plan)
{
	if( plan ) {
		delete plan;
	}
}

const char* dedisp_get_error_string(dedisp_error error)
{
	switch( error ) {
	case DEDISP_NO_ERROR:
		return "No error";
	case DEDISP_MEM_ALLOC_FAILED:
		return "Memory allocation failed";
	case DEDISP_MEM_COPY_FAILED:
		return "Memory copy failed";
	case DEDISP_INVALID_DEVICE_INDEX:
		return "Invalid device index";
	case DEDISP_DEVICE_ALREADY_SET:
		return "Device is already set and cannot be changed";
	case DEDISP_NCHANS_EXCEEDS_LIMIT:
		return "No. channels exceeds internal limit";
	case DEDISP_INVALID_PLAN:
		return "Invalid plan";
	case DEDISP_INVALID_POINTER:
		return "Invalid pointer";
	case DEDISP_INVALID_STRIDE:
		return "Invalid stride";
	case DEDISP_NO_DM_LIST_SET:
		return "No DM list has been set";
	case DEDISP_TOO_FEW_NSAMPS:
		return "No. samples < maximum delay";
	case DEDISP_INVALID_FLAG_COMBINATION:
		return "Invalid flag combination";
	case DEDISP_UNSUPPORTED_IN_NBITS:
		return "Unsupported in_nbits value";
	case DEDISP_UNSUPPORTED_OUT_NBITS:
		return "Unsupported out_nbits value";
	case DEDISP_PRIOR_GPU_ERROR:
		return "Prior GPU error.";
	case DEDISP_INTERNAL_GPU_ERROR:
		return "Internal GPU error. Please contact the author(s).";
	case DEDISP_UNKNOWN_ERROR:
		return "Unknown error. Please contact the author(s).";
	default:
		return "Invalid error code";
	}
}
// ----------------
