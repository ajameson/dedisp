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
  This file just contains crappy wrappers for CUDA memory functions
*/

#pragma once

typedef unsigned int gpu_size_t;

template<typename T>
bool malloc_device(T*& addr, gpu_size_t count) {
	cudaError_t error = cudaMalloc((void**)&addr, count*sizeof(T));
	if( error != cudaSuccess ) {
		return false;
	}
	return true;
}
template<typename T>
void free_device(T*& addr) {
	cudaFree(addr);
	addr = 0;
}
template<typename T>
bool copy_host_to_device(T* dst, const T* src,
						 gpu_size_t count, cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	// TODO: Passing a device pointer as src causes this to segfault
	cudaMemcpy/*Async*/(dst, src, count*sizeof(T),
						cudaMemcpyHostToDevice/*, stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
template<typename T>
bool copy_device_to_host(T* dst, const T* src,
						 gpu_size_t count, cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	cudaMemcpy/*Async*/(dst, src, count*sizeof(T),
						cudaMemcpyDeviceToHost/*, stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
#if 0
// ------- REMOVED --------
template<typename T>
bool copy_host_to_symbol(const char* symbol, const T* src,
						 gpu_size_t count, cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	cudaMemcpyToSymbol/*Async*/(symbol, src,
							count * sizeof(T),
							0, cudaMemcpyHostToDevice/*,
													   stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
template<typename U, typename T>
bool copy_device_to_symbol(/*const char**/U symbol, const T* src,
						   gpu_size_t count, cudaStream_t stream=0) {
	cudaMemcpyToSymbolAsync(symbol, src,
							count * sizeof(T),
							0, cudaMemcpyDeviceToDevice,
							stream);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
// ------- REMOVED --------
#endif
// Note: Strides must be given in units of bytes
template<typename T, typename U>
bool copy_host_to_device_2d(T* dst, gpu_size_t dst_stride,
                            const U* src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	cudaMemcpy2D/*Async*/(dst, dst_stride,//*sizeof(T),
	                      src, src_stride,//*sizeof(U),
						  width_bytes, height,
						  cudaMemcpyHostToDevice/*, stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) { return false; }
	//#endif
	return true;
}

template<typename T, typename U>
bool copy_device_to_host_2d(T* dst, gpu_size_t dst_stride,
                            const U* src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	cudaMemcpy2D/*Async*/(dst, dst_stride,
	                      src, src_stride,
						  width_bytes, height,
						  cudaMemcpyDeviceToHost/*, stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) { return false; }
	//#endif
	return true;
}

template<typename T, typename U>
bool copy_device_to_device_2d(T* dst, gpu_size_t dst_stride,
                              const U* src, gpu_size_t src_stride,
                              gpu_size_t width_bytes, gpu_size_t height,
                              cudaStream_t stream=0) {
	cudaMemcpy2D/*Async*/(dst, dst_stride,
	                      src, src_stride,
						  width_bytes, height,
						  cudaMemcpyDeviceToDevice/*, stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) { return false; }
	//#endif
	return true;
}
