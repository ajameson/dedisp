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
  dedisp.h
  By Ben Barsdell (2012)
  benbarsdell@gmail.com
  Originally developed while at the Centre for Astrophysics and Supercomputing,
  Swinburne University of Technology, Victoria, Australia
  
  Limitations:  in_nbits must be one of: 1, 2, 4, 8, 16, 32
               out_nbits must be one of: 8, 16, 32
*/

/*! \file dedisp.h
 *  \brief Defines the interface to the dedisp library
 *
 *  dedisp is a C library for computing the incoherent dedispersion transform
 *    using a GPU. It takes filterbank data as input and computes multiple
 *    dedispersed integrated time series at different dispersion measures.
 *  The application programming interface (API) is based on that of the FFTW
 *    library, and should be familiar to users of that package.
 *
 *  \bug Asynchronous execution is not currently operational.
 */

#ifndef DEDISP_H_INCLUDE_GUARD
#define DEDISP_H_INCLUDE_GUARD

// Use C linkage to allow cross-language use of the library
#ifdef __cplusplus
extern "C" {
#endif

// Types
// -----
typedef float                      dedisp_float;
typedef unsigned char              dedisp_byte;
typedef unsigned long              dedisp_size;
typedef int                        dedisp_bool;
typedef struct dedisp_plan_struct* dedisp_plan;

/*! \typedef dedisp_float
 * The floating-point data-type used by the library. This is currently
     guaranteed to be equivalent to 'float'.*/
/*! \typedef dedisp_byte
 * The byte data-type used by the library to store time-series data. */
/*! \typedef dedisp_size
 * The size data-type used by the library to store sizes/dimensions. */
/*! \typedef dedisp_bool
 * The boolean data-type used by the library. Note that this type is
     implementation-defined and may not be equivalent to 'bool'. */
/*! \typedef dedisp_plan
 * The plan type used by the library to reference a dedispersion plan.
     This is an opaque pointer type. */

// Flags
// -----
typedef enum {
	DEDISP_USE_DEFAULT       = 0,
	DEDISP_HOST_POINTERS     = 1 << 1,
	DEDISP_DEVICE_POINTERS   = 1 << 2,
	
	DEDISP_WAIT              = 1 << 3,
	DEDISP_ASYNC             = 1 << 4
} dedisp_flag;

/*! \enum dedisp_flag
 * Flags for the library:\n
 * DEDISP_USE_DEFAULT: Use the default settings.\n
 * DEDISP_HOST_POINTERS: Instruct the function that the given pointers point to memory on the host.\n
 * DEDISP_DEVICE_POINTERS: Instruct the function that the given pointers point to memory on the device.\n
 * DEDISP_WAIT: Instruct the function to wait until all device operations are complete before returning.\n
 * DEDISP_ASYNC: Instruct the function to return before all device operations are complete.
 */

// Error codes
// -----------
typedef enum {
	DEDISP_NO_ERROR,
	DEDISP_MEM_ALLOC_FAILED,
	DEDISP_MEM_COPY_FAILED,
	DEDISP_NCHANS_EXCEEDS_LIMIT,
	DEDISP_INVALID_PLAN,
	DEDISP_INVALID_POINTER,
	DEDISP_INVALID_STRIDE,
	DEDISP_NO_DM_LIST_SET,
	DEDISP_TOO_FEW_NSAMPS,
	DEDISP_INVALID_FLAG_COMBINATION,
	DEDISP_UNSUPPORTED_IN_NBITS,
	DEDISP_UNSUPPORTED_OUT_NBITS,
	DEDISP_INVALID_DEVICE_INDEX,
	DEDISP_DEVICE_ALREADY_SET,
	DEDISP_PRIOR_GPU_ERROR,
	DEDISP_INTERNAL_GPU_ERROR,
	DEDISP_UNKNOWN_ERROR
} dedisp_error;


/*! \enum dedisp_error
 * Error codes for the library:\n
 * DEDISP_NO_ERROR: No error occurred.\n
 * DEDISP_MEM_ALLOC_FAILED: A memory allocation failed.\n
 * DEDISP_MEM_COPY_FAILED: A memory copy failed. This is often due to one of the arrays passed to dedisp_execute being too small.\n
 * DEDISP_NCHANS_EXCEEDS_LIMIT: The number of channels exceeds the internal limit. The current limit is 8192.\n
 * DEDISP_INVALID_PLAN: The given plan is NULL.\n
 * DEDISP_INVALID_POINTER: A pointer is invalid, possibly NULL.\n
 * DEDISP_INVALID_STRIDE: A stride value is less than the corresponding dimension's size.\n
 * DEDISP_NO_DM_LIST_SET: No DM list has yet been set using either \ref dedisp_set_dm_list or \ref dedisp_generate_dm_list.\n
 * DEDISP_TOO_FEW_NSAMPS: The number of time samples is less than the maximum dedispersion delay.\n
 * DEDISP_INVALID_FLAG_COMBINATION: Some of the given flags are incompatible.\n
 * DEDISP_UNSUPPORTED_IN_NBITS: The given \p in_nbits value is not supported. See \ref dedisp_execute for supported values.\n
 * DEDISP_UNSUPPORTED_OUT_NBITS: The given \p out_nbits value is not supported. See \ref dedisp_execute for supported values.\n
 * DEDISP_INVALID_DEVICE_INDEX: The given device index does not correspond to a device in the system.\n
 * DEDISP_DEVICE_ALREADY_SET: The device has already been set and cannot be changed. See \ref dedisp_set_device for more info.\n
 * DEDISP_PRIOR_GPU_ERROR: There was an existing GPU error prior to calling the function.\n
 * DEDISP_INTERNAL_GPU_ERROR: An unexpected GPU error has occurred within the library. Please contact the authors if you get this error.\n
 * DEDISP_UNKNOWN_ERROR: An unexpected error has occurred. Please contact the authors if you get this error.
 */

// Plan management
// ---------------
/*! \p dedisp_create_plan builds a new plan object using the given parameters
 *  and returns it in \p *plan.
 *  
 *  \param plan Pointer to a dedisp_plan object
 *  \param nchans Number of frequency channels
 *  \param dt Time difference between two consecutive samples in seconds
 *  \param f0 Frequency of the first (i.e., highest frequency) channel in MHz
 *  \param df Frequency difference between two consecutive channels in MHz
 *  \return One of the following error codes: \n
 *  \p DEDISP_NO_ERROR, \p DEDISP_NCHANS_EXCEEDS_LIMIT,
 *  \p DEDISP_MEM_ALLOC_FAILED, \p DEDISP_MEM_COPY_FAILED,
 *  \p DEDISP_PRIOR_GPU_ERROR
 *  
 */
dedisp_error dedisp_create_plan(dedisp_plan* plan,
                                dedisp_size  nchans,
                                dedisp_float dt,
                                dedisp_float f0,
                                dedisp_float df);

/*! \p dedisp_destroy_plan frees a plan and its associated resources
 *  
 *  \param plan Plan object to destroy
 */ 
void         dedisp_destroy_plan(dedisp_plan plan);

// Setters
// -------
/*! \p dedisp_set_device sets the GPU device to be used by the library
 *
 *  \param device_idx The index of the device to use
 *  \return One of the following error codes: \n
 *  \p DEDISP_NO_ERROR, \p DEDISP_INVALID_DEVICE_INDEX,
 *  \p DEDISP_DEVICE_ALREADY_SET, \p DEDISP_UNKNOWN_ERROR, \p DEDISP_PRIOR_GPU_ERROR
 *  \note If this function is not called, the system-defined default device
 *    will be used.
 *  \note This function must be called <b>before any other dedisp library
 *    calls.</b>
 *  \note This function can only be called once.
 */
dedisp_error dedisp_set_device(int device_idx);

/*! \p dedisp_set_gulp_size sets the internal gulp size used by the library
 *  
 *  \param plan Plan object to set gulp size of
 *  \param gulp_size The new internal gulp size (arbitrary units, default: 131072)
 *  \return One of the following error codes: \n
 *  \p DEDISP_INVALID_PLAN, \p DEDISP_NO_ERROR
 *  \note If this function is not called, the default gulp size (131072)
 *          is used.
 */
dedisp_error dedisp_set_gulp_size(dedisp_plan plan,
                                  dedisp_size gulp_size);
/*! \p dedisp_get_gulp_size gets the internal gulp size used by the library
 *  
 *  \param plan Plan object to set gulp size of
 *  \return The internal gulp size used by the library (arbitrary units, default: 131072)
 */
dedisp_size  dedisp_get_gulp_size(dedisp_plan plan);
/*! \p dedisp_set_killmask sets a list of channels to ignore when dedispersing
 *
 *  \param plan Plan object to apply killmask to
 *  \param killmask Array containing one value per channel. 0=>ignore channel,
 *     1=>include channel.
 *  \return One of the following error codes: \n
 *  \p DEDISP_INVALID_PLAN, \p DEDISP_MEM_COPY_FAILED, \p DEDISP_PRIOR_GPU_ERROR,
 *  \p DEDISP_NO_ERROR
 *  \note \p killmask must be an array on the host, i.e., it must not be a
 *          device array
 *  \note Passing \p killmask=NULL sets a blank killmask, i.e., no channels
 *    are ignored.
 */
dedisp_error dedisp_set_killmask(dedisp_plan        plan,
                                 const dedisp_bool* killmask);

//dedisp_error dedisp_set_stream(dedisp_plan   plan,
//                               dedisp_stream stream);

/*! \p dedisp_set_dm_list sets a list of dispersion measures to be computed
 *       during dedispersion
 *  
 *  \param plan Plan object to apply dm list to.
 *  \param dm_list Array containing dispersion measures to be computed
 *    (in pc cm^-3).
 *  \param count Number of dispersion measures.
 *  \return One of the following error codes: \n
 *  \p DEDISP_NO_ERROR, \p DEDISP_INVALID_PLAN, \p DEDISP_INVALID_POINTER,
 *  \p DEDISP_MEM_ALLOC_FAILED, \p DEDISP_MEM_COPY_FAILED, \p DEDISP_PRIOR_GPU_ERROR
 *  \note \p dm_list must be an array on the host, i.e., it must not be a device array
 *  \note One of either this function or \p dedisp_generate_dm_list must be called
 *    prior to calling dedisp_execute.
 */
dedisp_error dedisp_set_dm_list(dedisp_plan         plan,
                                const dedisp_float* dm_list,
                                dedisp_size         count);
/*! \p dedisp_generate_dm_list generates a list of dispersion measures to be
 *       computed during dedispersion
 *  
 *  \param plan Plan object to generate DM list for.
 *  \param dm_start The lowest DM to use, in pc cm^-3.
 *  \param dm_end The highest DM to use, in pc cm^-3.
 *  \param pulse_width The expected intrinsic width of the pulse signal in 
 *    microseconds.
 *  \param tol The smearing tolerance factor between DM trials (a typical value
 *    is 1.25).
 *  \return One of the following error codes: \n
 *  \p DEDISP_NO_ERROR, \p DEDISP_INVALID_PLAN, \p DEDISP_MEM_ALLOC_FAILED,
 *  \p DEDISP_MEM_COPY_FAILED, \p DEDISP_PRIOR_GPU_ERROR
 *  \note The number of DMs generated can be obtained using
 *    \p dedisp_get_dm_count(plan).
 */
dedisp_error dedisp_generate_dm_list(dedisp_plan  plan,
                                     dedisp_float dm_start,
                                     dedisp_float dm_end,
                                     dedisp_float pulse_width,
                                     dedisp_float tol);

dedisp_float * dedisp_generate_dm_list_guru (dedisp_float dm_start, dedisp_float dm_end,
                                     double dt, double ti, double f0, double df,
                                     dedisp_size nchans, double tol, dedisp_size * dm_count);


// Getters
// -------
/*! \p dedisp_get_max_delay gets the maximum delay (in samples) applied during
 *  dedispersion. During dedispersion, the last max_delay samples are required
 *  for the computation of other samples, but are not themselves dedispersed.
 * 
 *  \return The maximum delay that is applied during execution of \p plan.
 *  \note A DM list for \p plan must exist prior to calling this function.
 */
dedisp_size         dedisp_get_max_delay(const dedisp_plan plan);
/*! \p dedisp_get_dm_delay gets the delay for a dispersion trial.
 * 
 *  \return The delay in samples for a given dispersion trial.
 */
dedisp_size         dedisp_get_dm_delay(const dedisp_plan plan, int channel);
/*! \p dedisp_get_channel_count gets the number of frequency channels in a plan.
 * 
 *  \return The number of frequency channels in \p plan.
 */
dedisp_size         dedisp_get_channel_count(const dedisp_plan plan);
/*! \p dedisp_get_dm_count gets the number of dispersion measures in a plan.
 * 
 *  \return The number of dispersion measures in \p plan.
 */
dedisp_size         dedisp_get_dm_count(const dedisp_plan plan);
/*! \p dedisp_get_dm_list gets the list of dispersion measures in a plan.
 * 
 *  \return Pointer to an array of dispersion measures in \p plan.
 *  \note This function returns a host pointer, not a device pointer.
 */
const dedisp_float* dedisp_get_dm_list(const dedisp_plan plan);
/*! \p dedisp_get_killmask gets the killmask used in a plan.
 * 
 *  \return Pointer to an array of killmask values in \p plan.
 *  \note This function returns a host pointer, not a device pointer.
 */
const dedisp_bool*  dedisp_get_killmask(const dedisp_plan plan);
/*! \p dedisp_get_dt gets the difference in time between two consecutive
 *    samples for a plan.
 * 
 *  \return The difference in time between two consecutive samples for \p plan.
 */
dedisp_float        dedisp_get_dt(const dedisp_plan plan);
/*! \p dedisp_get_f0 gets the frequency of the first channel for a plan.
 * 
 *  \return The frequency of the first channel for \p plan.
 */
dedisp_float        dedisp_get_f0(const dedisp_plan plan);
/*! \p dedisp_get_df gets the difference in frequency between two consecutive
 *    channels for a plan.
 * 
 *  \return The difference in frequency between two consecutive channels for
 *     \p plan.
 */
dedisp_float        dedisp_get_df(const dedisp_plan plan);
/*! \p dedisp_get_error_string gives a human-readable description of a
 *    given error code.
 * 
 *  \param error The error code to describe.
 *  \return A string describing the error code.
 */
const char*         dedisp_get_error_string(dedisp_error error);

// Plan execution
// --------------
/*! \p dedisp_execute executes a plan to dedisperse the given array of data.
 * 
 *  \param plan The plan to execute.
 *  \param nsamps The length in samples of each input time series.
 *  \param in Pointer to an array containing a time series of length \p nsamps
 *    for each frequency channel in \p plan.
 *    The data must be in <b>time-major order</b>, i.e., frequency is the 
 *      fastest-changing dimension, time the slowest.
 *    There must be no padding between consecutive frequency channels.
 *  \param in_nbits The number of bits per sample in the input data.
 *    Currently supported values are: 1, 2, 4, 8, 16, 32
 *  \param out Pointer to an output array of length
 *    (\p nsamps - \p max_delay) * \p dm_count * \p out_nbits/8 bytes, where
 *    \p max_delay and \p dm_count are the values given by
 *    \p dedisp_get_max_delay(plan) and \p dedisp_get_dm_count(plan).
 *    The output order is <b>DM-major</b>, i.e., time is the
 *    fastest-changing dimension, DM the slowest.
 *  \param out_nbits The number of bits per sample in the output data.
 *    Currently supported values are: 8 (uchar), 16 (ushort), 32 (float)
 *  \param flags Configuration flags, interpreted as follows: \n
 *    0 or DEDISP_USE_DEFAULT:        Use the default configuration. \n
 *    DEDISP_HOST_POINTERS (default): The pointers \p in and \p out point to
 *      arrays on the host. \n
 *    DEDISP_DEVICE_POINTERS:         The pointers \p in and \p out point to
 *      arrays on the device. \n
 *    DEDISP_WAIT (default): Force the function to wait until the
 *      computation is complete before returning. \n
 *    DEDISP_ASYNC: Allow the function to return before the computation is
 *      complete. The \p dedisp_sync function can later be used to wait
 *      until the computation is complete. \n
 *   Flags may be combined using the | operator.
 *  \return One of the following error codes: \n
 *    \p DEDISP_NO_ERROR, \p DEDISP_INVALID_PLAN, \p DEDISP_INVALID_POINTER,
 *    \p DEDISP_INVALID_STRIDE, \p DEDISP_NO_DM_LIST_SET, \p DEDISP_TOO_FEW_NSAMPS,
 *    \p DEDISP_INVALID_FLAG_COMBINATION, \p DEDISP_UNSUPPORTED_IN_NBITS,
 *    \p DEDISP_UNSUPPORTED_OUT_NBITS, \p DEDISP_INTERNAL_GPU_ERROR, \p DEDISP_PRIOR_GPU_ERROR
 *  \note This function assumes \p in and \p out are unpadded arrays. To process
 *  arrays containing padding, use \ref dedisp_execute_adv.
 *  \note This function assumes one wishes to compute all DMs in the DM list.
 *    To compute only a subset of the DM list, use \ref dedisp_execute_guru.
 */
dedisp_error dedisp_execute(const dedisp_plan  plan,
                            dedisp_size        nsamps,
                            const dedisp_byte* in,
                            dedisp_size        in_nbits,
                            dedisp_byte*       out,
                            dedisp_size        out_nbits,
                            unsigned           flags);
/*! \p dedisp_execute_adv executes a plan to dedisperse the given array of data.
 * 
 *  \param plan The plan to execute.
 *  \param nsamps The length in samples of each input time series.
 *  \param in Pointer to an array containing a time series of length \p nsamps
 *    for each frequency channel in \p plan.
 *    The data must be in <b>time-major order</b>, i.e., frequency is the 
 *      fastest-changing dimension, time the slowest.
 *  \param in_nbits The number of bits per sample in the input data.
 *    Currently supported values are: 1, 2, 4, 8, 16, 32
 *  \param in_stride The stride of the array \p in in bytes.
 *    Must be >= \p channel_count * \p in_nbits/8, where \p channel_count is the
 *      value given by \p dedisp_get_channel_count(plan).
 *  \param out Pointer to an array of length \p out_stride*dm_count bytes, where
 *    \p dm_count is the value given by \p dedisp_get_dm_count(plan).
 *    The output order is <b>DM-major</b>, i.e., time is the
 *    fastest-changing dimension, DM the slowest.
 *  \param out_nbits The number of bits per sample in the output data.
 *    Currently supported values are: 8 (uchar), 16 (ushort), 32 (float)
 *  \param out_stride The stride of the array \p out in bytes.
 *    Must be >= (\p nsamps - \p max_delay) * \p out_nbits/8, where \p max_delay
 *      is the value given by \p dedisp_get_max_delay(plan).
 *  \param flags Configuration flags, interpreted as follows: \n
 *    0 or DEDISP_USE_DEFAULT:        Use the default configuration. \n
 *    DEDISP_HOST_POINTERS (default): The pointers \p in and \p out point to
 *      arrays on the host. \n
 *    DEDISP_DEVICE_POINTERS:         The pointers \p in and \p out point to
 *      arrays on the device. \n
 *    DEDISP_WAIT (default): Force the function to wait until the
 *      computation is complete before returning. \n
 *    DEDISP_ASYNC: Allow the function to return before the computation is
 *      complete. The \p dedisp_sync function can later be used to wait
 *      until the computation is complete. \n
 *   Flags may be combined using the | operator.
 *  \return One of the following error codes: \n
 *    \p DEDISP_NO_ERROR, \p DEDISP_INVALID_PLAN, \p DEDISP_INVALID_POINTER,
 *    \p DEDISP_INVALID_STRIDE, \p DEDISP_NO_DM_LIST_SET, \p DEDISP_TOO_FEW_NSAMPS,
 *    \p DEDISP_INVALID_FLAG_COMBINATION, \p DEDISP_UNSUPPORTED_IN_NBITS,
 *    \p DEDISP_UNSUPPORTED_OUT_NBITS, \p DEDISP_INTERNAL_GPU_ERROR, \p DEDISP_PRIOR_GPU_ERROR
 *  \note This function assumes one wishes to compute all DMs in the DM list.
 *    To compute only a subset of the DM list, use \ref dedisp_execute_guru.
 */
dedisp_error dedisp_execute_adv(const dedisp_plan  plan,
                                dedisp_size        nsamps,
                                const dedisp_byte* in,
                                dedisp_size        in_nbits,
                                dedisp_size        in_stride,
                                dedisp_byte*       out,
                                dedisp_size        out_nbits,
                                dedisp_size        out_stride,
                                unsigned           flags);
/*! \p dedisp_execute_guru executes a plan to dedisperse the given array of data.
 *  \warning This function is experimental and may contain bugs.
 *  \bug This function cannot be used in conjunction with adaptive time
 *         resolution.
 * 
 *  \param plan The plan to execute.
 *  \param nsamps The length in samples of each input time series.
 *  \param in Pointer to an array containing a time series of length \p nsamps
 *    for each frequency channel in \p plan.
 *    The data must be in <b>time-major order</b>, i.e., frequency is the 
 *      fastest-changing dimension, time the slowest.
 *  \param in_nbits The number of bits per sample in the input data.
 *    Currently supported values are: 1, 2, 4, 8, 16, 32
 *  \param in_stride The stride of the array \p in in bytes.
 *    Must be >= \p channel_count, where \p channel_count is the value given by
 *      \p dedisp_get_channel_count(plan).
 *  \param out Pointer to an array of length \p out_stride*dm_count bytes, where
 *    \p dm_count is the value given by \p dedisp_get_dm_count(plan).
 *    The output order is <b>DM-major</b>, i.e., time is the
 *    fastest-changing dimension, DM the slowest.
 *  \param out_nbits The number of bits per sample in the output data.
 *    Currently supported values are: 8 (uchar), 16 (ushort), 32 (float)
 *  \param out_stride The stride of the array \p out in bytes.
 *    Must be >= (\p nsamps - \p max_delay) * \p out_nbits/8, where \p max_delay
 *      is the value given by \p dedisp_get_max_delay(plan).
 *  \param first_dm_idx The (0-based) index at which to start within the DM list.
 *  \param dm_count The number of DMs to compute from the DM list.
 *  \param flags Configuration flags, interpreted as follows: \n
 *    0 or DEDISP_USE_DEFAULT:        Use the default configuration. \n
 *    DEDISP_HOST_POINTERS (default): The pointers \p in and \p out point to
 *      arrays on the host. \n
 *    DEDISP_DEVICE_POINTERS:         The pointers \p in and \p out point to
 *      arrays on the device. \n
 *    DEDISP_WAIT (default): Force the function to wait until the
 *      computation is complete before returning. \n
 *    DEDISP_ASYNC: Allow the function to return before the computation is
 *      complete. The \p dedisp_sync function can later be used to wait
 *      until the computation is complete. \n
 *   Flags may be combined using the | operator.
 *  \return One of the following error codes: \n
 *    \p DEDISP_NO_ERROR, \p DEDISP_INVALID_PLAN, \p DEDISP_INVALID_POINTER,
 *    \p DEDISP_INVALID_STRIDE, \p DEDISP_NO_DM_LIST_SET, \p DEDISP_TOO_FEW_NSAMPS,
 *    \p DEDISP_INVALID_FLAG_COMBINATION, \p DEDISP_UNSUPPORTED_IN_NBITS,
 *    \p DEDISP_UNSUPPORTED_OUT_NBITS, \p DEDISP_INTERNAL_GPU_ERROR, \p DEDISP_PRIOR_GPU_ERROR
 */
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
                                 unsigned           flags);

// TODO: CHECK THE STATUS OF THIS FEATURE
/*! \p dedisp_sync waits until all previous plan executions have completed
 *       before returning. This function can be used in conjunction with
 *       the DEDISP_ASYNC flag in dedisp_execute to overlap computation on
 *       the host and the device.
 *  
 *  \return One of the following error codes: \n
 *  \p DEDISP_NO_ERROR, \p DEDISP_PRIOR_GPU_ERROR
*/
dedisp_error dedisp_sync(void);

/*! \p dedisp_enable_adaptive_dt instructs \p plan to use an adaptive
 *       time-resolution scheme. The time resolution is varied as a function
 *       of DM, decreasing by factors of 2 when the increase in smearing is
 *       below \p tol.
 * 
 *  \param plan The plan for which to enable adaptive time resolution.
 *  \param pulse_width The expected pulse width in microseconds (used to 
 *           determine the total smearing).
 *  \param tol The smearing tolerance at which the time resolution is reduced.
 *           A typical value is 1.15, meaning a tolerance of 15%.
 */
dedisp_error dedisp_enable_adaptive_dt(dedisp_plan  plan,
                                       dedisp_float pulse_width,
                                       dedisp_float tol);
/*! \p dedisp_disable_adaptive_dt disables adaptive time resolution for \p plan.
 *  \param plan The plan for which to disable adaptive time resolution.
 */
dedisp_error dedisp_disable_adaptive_dt(dedisp_plan plan);
/*! \p dedisp_using_adaptive_dt returns whether \p plan has adaptive time
 *       resolution enabled.
 *  \param plan The plan for which to query adaptive time resolution.
 */
dedisp_bool  dedisp_using_adaptive_dt(const dedisp_plan plan);
/*! \p dedisp_get_dt_factors returns an array of length
 *       \p dedisp_get_dm_count(\p plan) containing the integer factors by which
 *       the time resolution (1/dt) is decreased for each DM. Note that the
 *       factors are guaranteed to be monotonically increasing powers of two.
 *  \param plan The plan from which to get the values.
 *  \return A pointer to an array of integer factors.
 *  \note A DM list for \p plan must exist prior to calling this function.
 *  \note This function may safely be called even when 
 *          \p dedisp_enable_adaptive_dt has not been called; in such cases the
 *          returned list will consist entirely of ones.
 */
const dedisp_size* dedisp_get_dt_factors(const dedisp_plan plan);

#ifdef __cplusplus
} // closing brace for extern "C"
#endif

#endif // DEDISP_H_INCLUDE_GUARD
