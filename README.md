# Minimally Viable Examples Using the Rerun SDK in C++

This is primarily used in communications with the Rerun team

**NOTE**: Build flag `SKIP_IMG_LOG` turns the helper function `rr_log_mat_image` into a no-op.  This
was used to isolate our processes and rule them out as possible causes of the memory leak.

## Current Example:

- SDK Version: 0.22.1
- Viewer Version: 0.22.1
- Description:
    1. Condition #1 (control):
        * Build with `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKIP_IMG_LOG=true`
        * Run with `./build/rerun_cpp_mve --enable-rerun = false`
        * Result: stable memory consumption
    2. Condition #2:
        * Build with `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKIP_IMG_LOG=true`
        * Run with `./build/rerun_cpp_mve --enable-rerun = true`
        * Result: stable memory consumption
    3. Condition #3:
        * Build with `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKIP_IMG_LOG=false`
        * Run with `./build/rerun_cpp_mve --enable-rerun = false`
        * Result: stable memory consumption as long as `rerun::set_default_enabled(false)` prior to
            `RecordingStream` constructor
    2. Condition #4:
        * Build with `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKIP_IMG_LOG=false`
        * Run with `./build/rerun_cpp_mve --viewer_addr 192.168.1.50:9876 --enable-rerun = true`
        * **NOTE**: This connects to a remote viewer
        * Result: rapid increase in memory consumption for the sender
- Reference: [Discord Question: "alloc::raw_vec::finish_grow unbounded heap leak C++"](https://discord.com/channels/1062300748202921994/1380251130340315146)
